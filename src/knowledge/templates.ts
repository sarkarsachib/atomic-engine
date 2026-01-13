export interface CodeTemplate {
  id: string;
  name: string;
  version: string;
  tags: string[];
  files: Record<string, string>;
}

export const BUILTIN_TEMPLATES: CodeTemplate[] = [
  {
    id: 'tpl_node_express_postgres_v1',
    name: 'node-express-postgres',
    version: '1.0.0',
    tags: ['node', 'express', 'postgres', 'rest', 'mvc', 'monolith'],
    files: {
      'README.md': '# Node + Express + Postgres API\n',
      'src/index.ts': "console.log('Boot API');\n",
      'docker-compose.yml': "services:\n  db:\n    image: postgres:15\n"
    }
  },
  {
    id: 'tpl_nextjs_prisma_v1',
    name: 'nextjs-prisma',
    version: '1.0.0',
    tags: ['nextjs', 'react', 'prisma', 'postgres', 'web', 'serverless'],
    files: {
      'README.md': '# Next.js + Prisma\n',
      'src/pages/index.tsx': "export default function Home(){return <div>Hello</div>}\n"
    }
  },
  {
    id: 'tpl_fastapi_postgres_v1',
    name: 'fastapi-postgres',
    version: '1.0.0',
    tags: ['python', 'fastapi', 'postgres', 'api', 'mvc'],
    files: {
      'README.md': '# FastAPI + Postgres\n',
      'app/main.py': "from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/')\ndef read_root():\n    return {'ok': True}\n"
    }
  },
  {
    id: 'tpl_microservices_compose_v1',
    name: 'microservices-docker-compose',
    version: '1.0.0',
    tags: ['microservices', 'docker', 'compose', 'api', 'event-driven'],
    files: {
      'README.md': '# Microservices Skeleton\n',
      'docker-compose.yml': "services:\n  gateway:\n    build: ./gateway\n  users:\n    build: ./users\n  orders:\n    build: ./orders\n"
    }
  },
  {
    id: 'tpl_serverless_worker_v1',
    name: 'serverless-worker',
    version: '1.0.0',
    tags: ['serverless', 'worker', 'queue', 'event-driven'],
    files: {
      'README.md': '# Serverless Worker\n',
      'src/handler.ts': "export async function handler(){ return { ok: true }; }\n"
    }
  },
  {
    id: 'tpl_node_cli_v1',
    name: 'node-cli',
    version: '1.0.0',
    tags: ['node', 'cli', 'typescript'],
    files: {
      'README.md': '# Node CLI\n',
      'src/cli.ts': "console.log('cli');\n"
    }
  }
];
