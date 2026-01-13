import knowledgeSystem from './knowledge_system';

// Minimal Node http server wrapper (optional).
// This file exists to satisfy the "async API endpoints for indexing" requirement
/**
 * Create a minimal Node.js HTTP server that exposes a small set of knowledge-system endpoints without coupling to any web framework.
 *
 * Exposed routes:
 * - GET /health -> { ok: true }
 * - POST /knowledge/index -> accepts JSON body, enqueues an indexing job, responds with `{ accepted: true, jobId }`
 * - GET /knowledge/templates -> `{ templates: [...] }`
 * - GET /knowledge/metrics -> `{ dashboard: ... }`
 * - any other route -> 404 `{ error: 'not_found' }`
 *
 * @returns The created HTTP server instance
 */

export function createKnowledgeHttpServer(): unknown {
  const http = require('http');

  const server = http.createServer(async (req: any, res: any) => {
    const url: string = req.url ?? '/';

    if (req.method === 'GET' && url.startsWith('/health')) {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ ok: true }));
      return;
    }

    if (req.method === 'POST' && url.startsWith('/knowledge/index')) {
      const body = await readJson(req);
      // Body shape should match IndexProjectInput; in practice this would be called by the orchestrator.
      const { jobId } = knowledgeSystem.enqueueIndexProject(body);
      res.writeHead(202, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ accepted: true, jobId }));
      return;
    }

    if (req.method === 'GET' && url.startsWith('/knowledge/templates')) {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ templates: knowledgeSystem.listTemplates() }));
      return;
    }

    if (req.method === 'GET' && url.startsWith('/knowledge/metrics')) {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ dashboard: knowledgeSystem.metrics.getDashboard() }));
      return;
    }

    res.writeHead(404, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: 'not_found' }));
  });

  return server;
}

/**
 * Parse and return the JSON payload from an HTTP request body, falling back to an empty object on error.
 *
 * @param req - The incoming HTTP request stream whose body will be read and parsed as JSON
 * @returns The parsed object from the request body, or `{}` if the body is empty or cannot be parsed
 */
function readJson(req: any): Promise<any> {
  return new Promise(resolve => {
    let data = '';
    req.on('data', (chunk: any) => {
      data += String(chunk);
    });
    req.on('end', () => {
      try {
        resolve(JSON.parse(data || '{}'));
      } catch {
        resolve({});
      }
    });
  });
}