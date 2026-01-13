import knowledgeSystem from './knowledge_system';

// Minimal Node http server wrapper (optional).
// This file exists to satisfy the "async API endpoints for indexing" requirement
// without coupling the knowledge system to a specific web framework.

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
