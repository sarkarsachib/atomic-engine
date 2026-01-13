/**
 * Normalize an idea string for consistent hashing and comparison.
 *
 * Converts the input to lowercase, collapses consecutive whitespace into single spaces,
 * removes all characters except `a`–`z`, `0`–`9`, and space, and trims leading/trailing whitespace.
 *
 * @param text - The input string to normalize
 * @returns The normalized string
 */
export function normalizeIdeaText(text: string): string {
  return text
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .replace(/[^a-z0-9\s]/g, '')
    .trim();
}

/**
 * Compute a deterministic 32-bit FNV-1a hash of the given text and return it as an 8-character hexadecimal string.
 *
 * The input is normalized first by lowercasing, collapsing consecutive whitespace to a single space, removing characters
 * other than a–z, 0–9, and space, and trimming leading/trailing whitespace; the resulting string is what's hashed.
 *
 * @param text - The input text to normalize and hash
 * @returns An 8-character, zero-padded hexadecimal representation of the 32-bit FNV-1a hash of the normalized input
 */
export function hashText(text: string): string {
  const normalized = normalizeIdeaText(text);
  let hash = 0x811c9dc5;

  for (let i = 0; i < normalized.length; i++) {
    hash ^= normalized.charCodeAt(i);
    hash = (hash * 0x01000193) >>> 0;
  }

  return hash.toString(16).padStart(8, '0');
}