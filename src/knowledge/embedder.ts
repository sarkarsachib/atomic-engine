import { hashText } from './hash';

export type Vector = number[];

export interface Embedder {
  dimensions: number;
  embedText(text: string): Promise<Vector>;
}

export class DeterministicHashEmbedder implements Embedder {
  readonly dimensions: number;

  constructor(dimensions = 384) {
    this.dimensions = dimensions;
  }

  async embedText(text: string): Promise<Vector> {
    const vec = new Array<number>(this.dimensions).fill(0);
    const tokens = text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(t => t.length > 1);

    for (const token of tokens) {
      const h = parseInt(hashText(token), 16);
      const idx = h % this.dimensions;
      vec[idx] += 1;
    }

    // L2 normalize so dot(vecA, vecB) ~= cosine similarity
    let norm = 0;
    for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm) || 1;

    for (let i = 0; i < vec.length; i++) vec[i] = vec[i] / norm;

    return vec;
  }
}
