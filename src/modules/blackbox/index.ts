/**
 * Atomic BlackBox - Specification Generator
 */
import { Atom, Module } from '../../core/brain';

export interface Specification {
  title: string;
  description: string;
  objectives: string[];
  requirements: { functional: string[]; nonFunctional: string[] };
  architecture: { components: string[]; technologies: string[] };
  constraints: string[];
  timeline: string;
}

class AtomicBlackBox implements Module {
  name = 'blackbox';
  
  supports(atom: Atom): boolean {
    return atom.type === 'requirement' && atom.content.includes('Specification');
  }
  
  async generate(atoms: Atom[]): Promise<Specification> {
    console.log('[BlackBox] Generating specification...');
    const specAtom = atoms.find(a => a.content.includes('Specification'));
    const ideaText = specAtom?.content.replace('Specification: ', '') || '';
    
    return {
      title: this.generateTitle(ideaText),
      description: ideaText,
      objectives: ['Deliver functional solution', 'Meet user requirements', 'Ensure quality'],
      requirements: {
        functional: ['User authentication', 'Data management', 'Responsive UI'],
        nonFunctional: ['Scalable', 'Secure', 'Performant', 'Maintainable']
      },
      architecture: {
        components: ['Frontend', 'Backend', 'Database', 'API'],
        technologies: ['React', 'Node.js', 'PostgreSQL', 'Docker']
      },
      constraints: atoms.filter(a => a.type === 'constraint').map(a => a.content),
      timeline: 'TBD based on complexity'
    };
  }
  
  private generateTitle(text: string): string {
    return text.split(' ').slice(0, 6).map(w => 
      w.charAt(0).toUpperCase() + w.slice(1)
    ).join(' ');
  }
}

export const blackbox = new AtomicBlackBox();
export default blackbox;
