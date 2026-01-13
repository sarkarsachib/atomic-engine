/* Minimal ambient declarations to allow strict TypeScript builds without depending on @types/node.
   Runtime is Node.js, so these globals exist.
*/

declare const console: {
  log: (...args: unknown[]) => void;
  info: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  error: (...args: unknown[]) => void;
};

declare const process: {
  env: Record<string, string | undefined>;
};

declare function setTimeout(handler: (...args: unknown[]) => void, timeout?: number): unknown;
declare function clearTimeout(handle: unknown): void;

declare function setImmediate(handler: (...args: unknown[]) => void, ...args: unknown[]): void;

declare const require: (id: string) => any;

declare class Buffer extends Uint8Array {
  static from(input: string | ArrayBuffer | ArrayBufferView): Buffer;
}
