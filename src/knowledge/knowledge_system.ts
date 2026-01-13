import type { ParsedIdea } from '../core/parser';
import type { PackagedOutput } from '../assembly/packager';

import { DeterministicHashEmbedder } from './embedder';
import { hashText } from './hash';
import { InMemoryVectorStore, PgVectorStore, type VectorStore } from './vector_store';
import { InMemoryCache, RedisCache, type Cache } from './cache';
import { TemplateLibrary, type TemplateSelectionResult } from './template_library';
import { BUILTIN_TEMPLATES, type CodeTemplate } from './templates';
import { KnowledgeGraph, PatternRecognizer, type DetectedPattern } from './patterns';
import { KnowledgeMetrics } from './metrics';

export interface ProjectRecord {
  id: string;
  ideaHash: string;
  idea: string;
  parsed: ParsedIdea;
  selectedModules: string[];
  ragContext: string;
  outputs: Map<string, unknown>;
  packaged: PackagedOutput;
  createdAt: Date;
  durationMs?: number;
  templateId?: string;
  parentProjectId?: string;
  patterns: DetectedPattern[];
}

export interface DuplicateCheckResult {
  isDuplicate: boolean;
  project?: ProjectRecord;
  score?: number;
  reason?: 'hash' | 'semantic';
}

export interface IndexProjectInput {
  idea: string;
  parsed: ParsedIdea;
  selectedModules: string[];
  ragContext: string;
  outputs: Map<string, unknown>;
  packaged: PackagedOutput;
  durationMs?: number;
  parentProjectId?: string;
}

export interface RagContextInput {
  idea: string;
  parsed: ParsedIdea;
  selectedModules: string[];
}

export class KnowledgeSystem {
  readonly metrics = new KnowledgeMetrics();

  private readonly embedder = new DeterministicHashEmbedder(Number(process.env.ATOMIC_EMBED_DIM ?? 384));

  private readonly fallbackStore: VectorStore = new InMemoryVectorStore();
  private readonly primaryStore: VectorStore | undefined = process.env.DATABASE_URL
    ? new PgVectorStore(process.env.DATABASE_URL, this.embedder.dimensions)
    : undefined;

  private readonly store: { primaryOk: boolean } = { primaryOk: false };

  private readonly cache: Cache = process.env.REDIS_URL ? new RedisCache(process.env.REDIS_URL) : new InMemoryCache();

  private readonly templates: TemplateLibrary = new TemplateLibrary(this.embedder, this.fallbackStore, BUILTIN_TEMPLATES);
  private readonly patternRecognizer = new PatternRecognizer();
  private readonly graph = new KnowledgeGraph();

  private initPromise: Promise<void> | undefined;

  private projectsById = new Map<string, ProjectRecord>();
  private projectsByIdeaHash = new Map<string, string>();

  async checkDuplicateIdea(idea: string): Promise<DuplicateCheckResult> {
    await this.ensureInit();

    const ideaHash = hashText(idea);
    const existingId = this.projectsByIdeaHash.get(ideaHash);
    if (existingId) {
      const project = this.projectsById.get(existingId);
      if (project) {
        this.metrics.recordDuplicateAvoided();
        return { isDuplicate: true, project, score: 1, reason: 'hash' };
      }
    }

    const vector = await this.embedder.embedText(idea);
    const hits = await this.queryStore({ namespace: 'projects', vector, topK: 1, minScore: 0.93 });

    const best = hits[0];
    if (best) {
      const project = this.projectsById.get(best.id) ?? this.hydrateProject(best);
      if (project) {
        this.metrics.recordDuplicateAvoided();
        return { isDuplicate: true, project, score: best.score, reason: 'semantic' };
      }
    }

    return { isDuplicate: false };
  }

  async buildRagContext(input: RagContextInput): Promise<string> {
    await this.ensureInit();

    const key = `rag:${hashText(input.idea)}`;
    const cached = await this.cache.get(key);
    if (cached) return cached;

    const vector = await this.embedder.embedText(input.idea);
    const hits = await this.queryStore({ namespace: 'projects', vector, topK: 3, minScore: 0.55 });

    const lines: string[] = [];
    lines.push('RAG_CONTEXT');
    lines.push('---');

    if (hits.length === 0) {
      lines.push('No similar projects found yet.');
    } else {
      lines.push('Similar past projects:');
      for (const hit of hits) {
        const project = this.projectsById.get(hit.id);
        if (!project) continue;
        const arch = project.patterns.find(p => p.type === 'architecture')?.name;
        const stack = project.patterns.filter(p => p.type === 'stack').map(p => p.name).slice(0, 5);
        lines.push(`- ${project.id} (score ${hit.score.toFixed(2)}): ${project.idea}`);
        if (arch) lines.push(`  - architecture: ${arch}`);
        if (stack.length) lines.push(`  - stack: ${stack.join(', ')}`);
        if (project.templateId) lines.push(`  - template: ${project.templateId}`);
      }
    }

    const ctx = lines.join('\n').slice(0, 4000);
    await this.cache.set(key, ctx, 60 * 60);
    return ctx;
  }

  async selectTemplateForIdea(input: { idea: string; keywords: string[] }): Promise<TemplateSelectionResult> {
    await this.ensureInit();

    const base = await this.templates.selectBest(input, { minScore: 0.2, topK: 5 });
    if (base.alternatives.length <= 1) return base;

    const metrics = this.metrics.getTemplateMetrics();

    const scored = base.alternatives.map((tpl, idx) => {
      const m = metrics[tpl.id];
      const attempted = (m?.succeeded ?? 0) + (m?.failed ?? 0);
      const successRate = attempted > 0 ? (m!.succeeded ?? 0) / attempted : 0.5;
      const semanticRank = (base.alternatives.length - idx) / base.alternatives.length;
      const score = semanticRank * 0.8 + successRate * 0.2;
      return { tpl, score, successRate };
    });

    scored.sort((a, b) => b.score - a.score);

    const selected = scored[0]?.tpl;
    const recommendationNotes = [
      ...base.recommendationNotes,
      'Re-ranked templates using historical success rate.'
    ];

    return {
      selected,
      alternatives: scored.map(s => s.tpl),
      recommendationNotes
    };
  }

  async indexProject(input: IndexProjectInput): Promise<ProjectRecord> {
    await this.ensureInit();

    const now = new Date();
    const id = this.generateProjectId();
    const ideaHash = hashText(input.idea);

    const templateId = this.extractTemplateId(input.outputs);

    const patterns = [
      ...this.patternRecognizer.detectArchitecture({ idea: input.idea, parsed: input.parsed, files: input.packaged.files }),
      ...this.patternRecognizer.detectTechStack(input.parsed),
      ...this.patternRecognizer.detectAntiPatterns({ parsed: input.parsed, files: input.packaged.files })
    ];

    const record: ProjectRecord = {
      id,
      ideaHash,
      idea: input.idea,
      parsed: input.parsed,
      selectedModules: input.selectedModules,
      ragContext: input.ragContext,
      outputs: input.outputs,
      packaged: input.packaged,
      createdAt: now,
      durationMs: input.durationMs,
      templateId,
      parentProjectId: input.parentProjectId,
      patterns
    };

    this.projectsById.set(id, record);
    this.projectsByIdeaHash.set(ideaHash, id);

    for (const p of patterns) {
      this.graph.addEdge({ from: id, to: p.id, type: p.type, weight: p.confidence });
    }

    // Index embeddings in both stores (local + pgvector if available).
    // We embed a compact "project fingerprint" rather than only the idea text.
    const embeddingText = this.buildProjectEmbeddingText({
      idea: input.idea,
      parsed: input.parsed,
      outputs: input.outputs,
      templateId
    });
    const vector = await this.embedder.embedText(embeddingText);

    await this.upsertStores([
      {
        id,
        namespace: 'projects',
        text: embeddingText,
        vector,
        metadata: {
          createdAt: now.toISOString(),
          ideaHash,
          idea: input.idea,
          selectedModules: input.selectedModules,
          templateId,
          parentProjectId: input.parentProjectId,
          durationMs: input.durationMs,
          patterns: patterns.map(p => ({ id: p.id, type: p.type, name: p.name, confidence: p.confidence }))
        },
        createdAt: now
      }
    ]);

    const isRerun = Boolean(input.parentProjectId);
    this.metrics.recordProjectIndexed({ isRerun });
    if (templateId) this.metrics.recordTemplateOutcome(templateId, true);

    return record;
  }

  enqueueIndexProject(input: IndexProjectInput): { jobId: string } {
    const jobId = `job_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
    setImmediate(() => {
      this.indexProject(input).catch(err => console.error('[KnowledgeSystem] async index failed', err));
    });
    return { jobId };
  }

  getProject(id: string): ProjectRecord | undefined {
    return this.projectsById.get(id);
  }

  listTemplates(): CodeTemplate[] {
    return this.templates.list();
  }

  getKnowledgeGraph(): KnowledgeGraph {
    return this.graph;
  }

  private async ensureInit(): Promise<void> {
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      // Determine whether pgvector is available.
      if (this.primaryStore && (await this.primaryStore.healthCheck()) === 'ok') {
        this.store.primaryOk = true;
      }

      // Always warm in-memory template embeddings for fast selection.
      await this.templates.warm();

      // If pgvector is healthy, mirror templates for cross-process retrieval.
      if (this.store.primaryOk && this.primaryStore) {
        const now = new Date();
        const records = await Promise.all(
          this.templates.list().map(async tpl => ({
            id: tpl.id,
            namespace: 'templates',
            text: `${tpl.name} ${tpl.tags.join(' ')}`,
            vector: await this.embedder.embedText(`${tpl.name} ${tpl.tags.join(' ')}`),
            metadata: { name: tpl.name, version: tpl.version, tags: tpl.tags },
            createdAt: now
          }))
        );
        await this.primaryStore.upsert(records);
      }
    })();

    return this.initPromise;
  }

  private async upsertStores(records: any[]): Promise<void> {
    await this.fallbackStore.upsert(records);
    if (this.store.primaryOk && this.primaryStore) {
      try {
        await this.primaryStore.upsert(records);
      } catch {
        this.store.primaryOk = false;
      }
    }
  }

  private async queryStore(query: any): Promise<any[]> {
    if (this.store.primaryOk && this.primaryStore) {
      try {
        return await this.primaryStore.query(query);
      } catch {
        this.store.primaryOk = false;
      }
    }

    return this.fallbackStore.query(query);
  }

  private hydrateProject(hit: {
    id: string;
    metadata?: Record<string, unknown>;
    text?: string;
  }): ProjectRecord | undefined {
    const meta = hit.metadata ?? {};
    const idea = (typeof meta.idea === 'string' ? (meta.idea as string) : hit.text) ?? '';
    if (!idea) return undefined;

    const createdAt = typeof meta.createdAt === 'string' ? new Date(meta.createdAt as string) : new Date();
    const selectedModules = Array.isArray(meta.selectedModules) ? (meta.selectedModules as string[]) : [];

    const packaged: PackagedOutput = {
      structure: new Map<string, unknown>(),
      files: new Map<string, string>(),
      metadata: {
        generatedAt: createdAt,
        modules: selectedModules,
        totalFiles: 0
      }
    };

    const patternsRaw = Array.isArray(meta.patterns) ? (meta.patterns as any[]) : [];
    const patterns: DetectedPattern[] = patternsRaw
      .map(p => ({
        id: typeof p?.id === 'string' ? p.id : 'unknown',
        type: p?.type === 'architecture' || p?.type === 'stack' || p?.type === 'anti_pattern' ? p.type : 'stack',
        name: typeof p?.name === 'string' ? p.name : 'Unknown',
        confidence: typeof p?.confidence === 'number' ? p.confidence : 0.5
      }))
      .slice(0, 50);

    const record: ProjectRecord = {
      id: hit.id,
      ideaHash: hashText(idea),
      idea,
      parsed: {
        intent: 'unknown',
        features: [],
        constraints: [],
        targets: [],
        priority: 'medium',
        complexity: 5,
        keywords: []
      },
      selectedModules,
      ragContext: '',
      outputs: new Map<string, unknown>(),
      packaged,
      createdAt,
      templateId: typeof meta.templateId === 'string' ? (meta.templateId as string) : undefined,
      patterns
    };

    this.projectsById.set(record.id, record);
    this.projectsByIdeaHash.set(record.ideaHash, record.id);

    return record;
  }

  private buildProjectEmbeddingText(input: {
    idea: string;
    parsed: ParsedIdea;
    outputs: Map<string, unknown>;
    templateId?: string;
  }): string {
    const keywords = input.parsed.keywords.slice(0, 10).join(' ');

    const spec = input.outputs.get('blackbox');
    const specText = spec && typeof spec === 'object' ? JSON.stringify(spec) : '';

    const forgeOut = input.outputs.get('forge');
    const forgeText = forgeOut && typeof forgeOut === 'object' ? JSON.stringify(forgeOut) : '';

    const parts = [
      `idea: ${input.idea}`,
      `intent: ${input.parsed.intent}`,
      `keywords: ${keywords}`,
      input.templateId ? `template: ${input.templateId}` : '',
      specText ? `spec: ${specText}` : '',
      forgeText ? `code: ${forgeText}` : ''
    ].filter(Boolean);

    // Keep the text bounded to avoid oversized payloads.
    return parts.join('\n').slice(0, 8000);
  }

  private extractTemplateId(outputs: Map<string, unknown>): string | undefined {
    const forgeOut = outputs.get('forge');
    if (!forgeOut || typeof forgeOut !== 'object') return undefined;
    const tpl = (forgeOut as any).template;
    if (tpl && typeof tpl.id === 'string') return tpl.id;
    return undefined;
  }

  private generateProjectId(): string {
    return `proj_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
  }
}

export const knowledgeSystem = new KnowledgeSystem();
export default knowledgeSystem;
