export interface Cache {
  get(key: string): Promise<string | undefined>;
  set(key: string, value: string, ttlSeconds: number): Promise<void>;
}

export class InMemoryCache implements Cache {
  private store = new Map<string, { value: string; expiresAt: number }>();

  async get(key: string): Promise<string | undefined> {
    const item = this.store.get(key);
    if (!item) return undefined;
    if (Date.now() > item.expiresAt) {
      this.store.delete(key);
      return undefined;
    }
    return item.value;
  }

  async set(key: string, value: string, ttlSeconds: number): Promise<void> {
    this.store.set(key, { value, expiresAt: Date.now() + ttlSeconds * 1000 });
  }
}

export class RedisCache implements Cache {
  private client: any | undefined;

  constructor(redisUrl: string) {
    try {
      const redis = require('redis');
      this.client = redis.createClient({ url: redisUrl });
      this.client.on('error', () => undefined);
      this.client.connect?.();
    } catch {
      this.client = undefined;
    }
  }

  async get(key: string): Promise<string | undefined> {
    if (!this.client) return undefined;
    try {
      const v = await this.client.get(key);
      return typeof v === 'string' ? v : undefined;
    } catch {
      return undefined;
    }
  }

  async set(key: string, value: string, ttlSeconds: number): Promise<void> {
    if (!this.client) return;
    try {
      await this.client.set(key, value, { EX: ttlSeconds });
    } catch {
      // ignore
    }
  }
}
