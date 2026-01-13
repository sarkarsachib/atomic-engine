import type { ParsedIdea } from '../core/parser';

export type ArchitectureType =
  | 'MVC'
  | 'Microservices'
  | 'Serverless'
  | 'EventDriven'
  | 'CLI'
  | 'Monolith';

export interface DetectedPattern {
  id: string;
  type: 'architecture' | 'stack' | 'anti_pattern';
  name: string;
  confidence: number;
  details?: Record<string, unknown>;
}

export class PatternRecognizer {
  detectArchitecture(input: { idea: string; parsed: ParsedIdea; files: Map<string, string> }): DetectedPattern[] {
    const text = `${input.idea} ${input.parsed.keywords.join(' ')}`.toLowerCase();
    const filePaths = Array.from(input.files.keys()).join(' ').toLowerCase();

    const patterns: DetectedPattern[] = [];

    const add = (name: ArchitectureType, confidence: number): void => {
      patterns.push({
        id: `arch_${name.toLowerCase()}`,
        type: 'architecture',
        name,
        confidence
      });
    };

    if (text.includes('microservice') || filePaths.includes('docker-compose') || filePaths.includes('gateway')) {
      add('Microservices', 0.85);
    }

    if (text.includes('serverless') || text.includes('lambda') || text.includes('vercel') || filePaths.includes('vercel')) {
      add('Serverless', 0.75);
    }

    if (text.includes('event') || text.includes('queue') || text.includes('kafka') || text.includes('rabbit')) {
      add('EventDriven', 0.7);
    }

    if (text.includes('cli') || filePaths.includes('cli.ts')) {
      add('CLI', 0.7);
    }

    if (filePaths.includes('controllers') || filePaths.includes('models') || filePaths.includes('routes')) {
      add('MVC', 0.7);
    }

    if (patterns.length === 0) add('Monolith', 0.6);

    return patterns;
  }

  detectTechStack(parsed: ParsedIdea): DetectedPattern[] {
    const text = parsed.keywords.join(' ').toLowerCase();
    const stack = new Set<string>();

    const maybe = (token: string, key: string): void => {
      if (text.includes(token)) stack.add(key);
    };

    maybe('react', 'React');
    maybe('next', 'Next.js');
    maybe('node', 'Node.js');
    maybe('express', 'Express');
    maybe('postgres', 'PostgreSQL');
    maybe('redis', 'Redis');
    maybe('docker', 'Docker');
    maybe('python', 'Python');
    maybe('fastapi', 'FastAPI');

    return Array.from(stack).map(name => ({
      id: `stack_${name.toLowerCase().replace(/[^a-z0-9]/g, '_')}`,
      type: 'stack',
      name,
      confidence: 0.6
    }));
  }

  detectAntiPatterns(input: { parsed: ParsedIdea; files: Map<string, string> }): DetectedPattern[] {
    const patterns: DetectedPattern[] = [];

    const hasTests = Array.from(input.files.keys()).some(p => p.includes('test') || p.includes('__tests__'));
    if (!hasTests) {
      patterns.push({
        id: 'anti_missing_tests',
        type: 'anti_pattern',
        name: 'MissingTests',
        confidence: 0.7,
        details: { suggestion: 'Add a minimal test harness early to prevent regressions.' }
      });
    }

    if (input.parsed.complexity >= 8 && input.parsed.features.length === 0) {
      patterns.push({
        id: 'anti_underspecified',
        type: 'anti_pattern',
        name: 'UnderspecifiedRequirements',
        confidence: 0.6,
        details: { suggestion: 'Clarify functional requirements; high complexity with few explicit features.' }
      });
    }

    return patterns;
  }
}

export interface GraphEdge {
  from: string;
  to: string;
  type: string;
  weight: number;
}

export class KnowledgeGraph {
  private readonly edges: GraphEdge[] = [];

  addEdge(edge: GraphEdge): void {
    this.edges.push(edge);
  }

  listEdges(): GraphEdge[] {
    return [...this.edges];
  }
}
