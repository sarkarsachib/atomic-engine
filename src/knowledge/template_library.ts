import type { Embedder } from './embedder';
import type { VectorStore } from './vector_store';
import type { CodeTemplate } from './templates';

export interface TemplateSelectionInput {
  idea: string;
  keywords: string[];
}

export interface TemplateSelectionResult {
  selected?: CodeTemplate;
  alternatives: CodeTemplate[];
  recommendationNotes: string[];
}

export class TemplateLibrary {
  private readonly templates = new Map<string, CodeTemplate>();

  constructor(
    private readonly embedder: Embedder,
    private readonly vectorStore: VectorStore,
    initialTemplates: CodeTemplate[]
  ) {
    for (const tpl of initialTemplates) {
      this.templates.set(tpl.id, tpl);
    }
  }

  list(): CodeTemplate[] {
    return Array.from(this.templates.values());
  }

  get(id: string): CodeTemplate | undefined {
    return this.templates.get(id);
  }

  async warm(): Promise<void> {
    const now = new Date();
    const records = [];
    for (const tpl of this.templates.values()) {
      const text = `${tpl.name} ${tpl.tags.join(' ')}`;
      const vec = await this.embedder.embedText(text);
      records.push({
        id: tpl.id,
        namespace: 'templates',
        text,
        vector: vec,
        metadata: { name: tpl.name, version: tpl.version, tags: tpl.tags },
        createdAt: now
      });
    }

    await this.vectorStore.upsert(records);
  }

  async selectBest(input: TemplateSelectionInput, opts?: { minScore?: number; topK?: number }): Promise<TemplateSelectionResult> {
    const queryText = `${input.idea} ${input.keywords.join(' ')}`;
    const vector = await this.embedder.embedText(queryText);

    const hits = await this.vectorStore.query({
      namespace: 'templates',
      vector,
      topK: opts?.topK ?? 5,
      minScore: opts?.minScore ?? 0.2
    });

    const alternatives: CodeTemplate[] = [];
    for (const hit of hits) {
      const tpl = this.templates.get(hit.id);
      if (tpl) alternatives.push(tpl);
    }

    const selected = alternatives[0];

    const notes: string[] = [];
    if (selected) {
      notes.push(`Selected template based on semantic match (score ~ ${hits[0]?.score.toFixed(2) ?? 'n/a'}).`);
      notes.push(`Tags: ${selected.tags.join(', ')}`);
    } else {
      notes.push('No strong template match found; using a generic scaffold.');
    }

    return {
      selected,
      alternatives,
      recommendationNotes: notes
    };
  }
}
