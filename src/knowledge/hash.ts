export function normalizeIdeaText(text: string): string {
  return text
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .replace(/[^a-z0-9\s]/g, '')
    .trim();
}

// FNV-1a 32-bit
export function hashText(text: string): string {
  const normalized = normalizeIdeaText(text);
  let hash = 0x811c9dc5;

  for (let i = 0; i < normalized.length; i++) {
    hash ^= normalized.charCodeAt(i);
    hash = (hash * 0x01000193) >>> 0;
  }

  return hash.toString(16).padStart(8, '0');
}
