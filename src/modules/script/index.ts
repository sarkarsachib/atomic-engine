/**
 * Atomic Script - Documentation Generator
 */
import { Atom, Module } from '../../core/types';

export interface DocumentationOutput {
  docs: string;
  sections: Record<string, string>;
}

class AtomicScript implements Module {
  name = 'script' as const;

  supports(atom: Atom): boolean {
    return atom.modules.includes(this.name);
  }

  async generate(atoms: Atom[]): Promise<DocumentationOutput> {
    const idea = atoms.find(a => a.type === 'raw')?.content ?? '';
    const context = atoms.find(a => a.type === 'context')?.content ?? '';

    const sections: Record<string, string> = {
      overview: `## Overview\n\n${idea}\n`,
      rag_context: context ? `## Retrieved Context\n\n${context}\n` : '## Retrieved Context\n\nNone\n'
    };

    return {
      docs: Object.values(sections).join('\n'),
      sections
    };
  }
}

export const script = new AtomicScript();
export default script;
