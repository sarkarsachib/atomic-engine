import type { Vector } from './embedder';

export interface VectorRecord {
  id: string;
  namespace: string;
  text: string;
  vector: Vector;
  metadata?: Record<string, unknown>;
  createdAt: Date;
}

export interface VectorQuery {
  namespace: string;
  vector: Vector;
  topK: number;
  minScore?: number;
}

export interface VectorQueryResult {
  id: string;
  score: number;
  metadata?: Record<string, unknown>;
  text?: string;
}

export interface VectorStore {
  upsert(records: VectorRecord[]): Promise<void>;
  query(query: VectorQuery): Promise<VectorQueryResult[]>;
  getById(namespace: string, id: string): Promise<VectorRecord | undefined>;
  healthCheck(): Promise<'ok' | 'unavailable'>;
}

function toFloat32(v: Vector): Float32Array {
  return new Float32Array(v);
}

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) sum += a[i] * b[i];
  return sum;
}

export class InMemoryVectorStore implements VectorStore {
  private recordsByNamespace: Map<string, Map<string, VectorRecord>> = new Map();
  private vectorsByNamespace: Map<string, Map<string, Float32Array>> = new Map();

  async upsert(records: VectorRecord[]): Promise<void> {
    for (const record of records) {
      const byId = this.recordsByNamespace.get(record.namespace) ?? new Map<string, VectorRecord>();
      const byVec = this.vectorsByNamespace.get(record.namespace) ?? new Map<string, Float32Array>();
      byId.set(record.id, record);
      byVec.set(record.id, toFloat32(record.vector));
      this.recordsByNamespace.set(record.namespace, byId);
      this.vectorsByNamespace.set(record.namespace, byVec);
    }
  }

  async query(query: VectorQuery): Promise<VectorQueryResult[]> {
    const vectors = this.vectorsByNamespace.get(query.namespace);
    const records = this.recordsByNamespace.get(query.namespace);

    if (!vectors || !records) return [];

    const q = toFloat32(query.vector);

    const results: VectorQueryResult[] = [];
    for (const [id, v] of vectors.entries()) {
      const score = dot(q, v);
      if (query.minScore !== undefined && score < query.minScore) continue;
      const record = records.get(id);
      results.push({ id, score, metadata: record?.metadata, text: record?.text });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, query.topK);
  }

  async getById(namespace: string, id: string): Promise<VectorRecord | undefined> {
    return this.recordsByNamespace.get(namespace)?.get(id);
  }

  async healthCheck(): Promise<'ok' | 'unavailable'> {
    return 'ok';
  }
}

// Optional pgvector-backed store. Will gracefully degrade if the pg client is not installed.
export class PgVectorStore implements VectorStore {
  private pool: any | undefined;
  private initialized = false;

  constructor(private readonly connectionString: string, private readonly dimensions: number) {
    try {
      const pg = require('pg');
      this.pool = new pg.Pool({ connectionString });
    } catch (_err) {
      this.pool = undefined;
    }
  }

  async upsert(records: VectorRecord[]): Promise<void> {
    if (!this.pool) return;
    await this.ensureInit();

    const client = await this.pool.connect();
    try {
      for (const r of records) {
        await client.query(
          `INSERT INTO atomic_knowledge_vectors (namespace, id, text, embedding, metadata, created_at)
           VALUES ($1, $2, $3, $4, $5, $6)
           ON CONFLICT (namespace, id) DO UPDATE SET
             text = EXCLUDED.text,
             embedding = EXCLUDED.embedding,
             metadata = EXCLUDED.metadata`,
          [r.namespace, r.id, r.text, this.vectorToSql(r.vector), JSON.stringify(r.metadata ?? {}), r.createdAt]
        );
      }
    } finally {
      client.release();
    }
  }

  async query(query: VectorQuery): Promise<VectorQueryResult[]> {
    if (!this.pool) return [];
    await this.ensureInit();

    const client = await this.pool.connect();
    try {
      const rows = await client.query(
        `SELECT id, text, metadata,
                1 - (embedding <=> $3) AS score
         FROM atomic_knowledge_vectors
         WHERE namespace = $1
         ORDER BY embedding <=> $3
         LIMIT $2`,
        [query.namespace, query.topK, this.vectorToSql(query.vector)]
      );

      return (rows.rows as any[])
        .map(r => ({
          id: String(r.id),
          score: Number(r.score),
          metadata: this.safeJson(r.metadata),
          text: String(r.text)
        }))
        .filter(r => (query.minScore === undefined ? true : r.score >= query.minScore));
    } finally {
      client.release();
    }
  }

  async getById(namespace: string, id: string): Promise<VectorRecord | undefined> {
    if (!this.pool) return undefined;
    await this.ensureInit();

    const rows = await this.pool.query(
      `SELECT namespace, id, text, embedding, metadata, created_at
       FROM atomic_knowledge_vectors
       WHERE namespace = $1 AND id = $2`,
      [namespace, id]
    );

    const row = rows.rows?.[0];
    if (!row) return undefined;

    return {
      namespace: String(row.namespace),
      id: String(row.id),
      text: String(row.text),
      vector: this.sqlToVector(String(row.embedding)),
      metadata: this.safeJson(row.metadata),
      createdAt: new Date(String(row.created_at))
    };
  }

  async healthCheck(): Promise<'ok' | 'unavailable'> {
    if (!this.pool) return 'unavailable';
    try {
      await this.pool.query('SELECT 1');
      return 'ok';
    } catch (_err) {
      return 'unavailable';
    }
  }

  private async ensureInit(): Promise<void> {
    if (!this.pool || this.initialized) return;

    // Opt-in auto-migration.
    if (process.env.ATOMIC_KNOWLEDGE_AUTO_MIGRATE !== '1') {
      this.initialized = true;
      return;
    }

    await this.pool.query('CREATE EXTENSION IF NOT EXISTS vector');
    await this.pool.query(
      `CREATE TABLE IF NOT EXISTS atomic_knowledge_vectors (
        namespace text NOT NULL,
        id text NOT NULL,
        text text NOT NULL,
        embedding vector(${this.dimensions}) NOT NULL,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (namespace, id)
      )`
    );
    await this.pool.query(
      `CREATE INDEX IF NOT EXISTS atomic_knowledge_vectors_embedding_idx
       ON atomic_knowledge_vectors USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)`
    );

    this.initialized = true;
  }

  private vectorToSql(vector: Vector): string {
    const padded = vector.slice(0, this.dimensions);
    while (padded.length < this.dimensions) padded.push(0);
    return `[${padded.join(',')}]`;
  }

  private sqlToVector(raw: string): Vector {
    const cleaned = raw.replace(/^[\[]|[\]]$/g, '');
    if (!cleaned.trim()) return [];
    return cleaned.split(',').map(v => Number(v.trim()));
  }

  private safeJson(value: unknown): Record<string, unknown> {
    if (!value) return {};
    if (typeof value === 'object') return value as Record<string, unknown>;
    try {
      return JSON.parse(String(value)) as Record<string, unknown>;
    } catch {
      return {};
    }
  }
}
