export {};

declare global {
  const process: {
    env: Record<string, string | undefined>;
  };

  function setImmediate(handler: (...args: unknown[]) => void, ...args: unknown[]): void;

  const require: (id: string) => any;
}
