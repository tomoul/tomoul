#!/usr/bin/env node
/**
 * Qwen3.5-0.8B Chat Server
 *
 * Zero-dependency Node.js server for the Tomoul Qwen3.5 chat demo.
 * Spawns the native binary for inference and serves a web UI.
 *
 * Usage:
 *   node examples/qwen3_5-chat/server.js
 *   node examples/qwen3_5-chat/server.js --port 8080
 *   node examples/qwen3_5-chat/server.js --weights artifacts/qwen3_5_0.8b_f32.tl
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const PROJECT_ROOT = path.join(__dirname, '../..');

const defaults = {
  port: 3000,
  binary: path.join(PROJECT_ROOT, 'zig-out/bin/tomoul_qwen3_5-0.8b.exe'),
  weights: path.join(PROJECT_ROOT, 'artifacts/qwen3_5_0.8b_q8k.tl'),
  tokenizer: path.join(PROJECT_ROOT, 'artifacts/qwen3_5_vocab.bin'),
  maxTokens: 256,
};

// Parse CLI args
const args = process.argv.slice(2);
const cfg = { ...defaults };
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--port' && args[i + 1]) cfg.port = parseInt(args[++i], 10);
  if (args[i] === '--weights' && args[i + 1]) cfg.weights = args[++i];
  if (args[i] === '--tokenizer' && args[i + 1]) cfg.tokenizer = args[++i];
  if (args[i] === '--binary' && args[i + 1]) cfg.binary = args[++i];
  if (args[i] === '--max-tokens' && args[i + 1]) cfg.maxTokens = parseInt(args[++i], 10);
}

// Validate files exist
for (const [label, fp] of [['binary', cfg.binary], ['weights', cfg.weights], ['tokenizer', cfg.tokenizer]]) {
  if (!fs.existsSync(fp)) {
    console.error(`[ERROR] ${label} not found: ${fp}`);
    process.exit(1);
  }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/**
 * Run inference and return { text, tokens, elapsed_ms, tok_per_s }.
 * The tomoul binary writes everything to stderr via std.debug.print.
 */
function generate(prompt, maxTokens = cfg.maxTokens) {
  return new Promise((resolve, reject) => {
    const child = spawn(cfg.binary, [
      'generate', prompt,
      '-w', cfg.weights,
      '-t', cfg.tokenizer,
      '-n', String(maxTokens),
    ]);

    let stderr = '';
    child.stderr.on('data', (d) => { stderr += d.toString(); });
    child.stdout.on('data', (d) => { stderr += d.toString(); }); // safety

    child.on('error', (err) => reject(err));
    child.on('close', (code) => {
      if (code !== 0) return reject(new Error(`Process exited ${code}: ${stderr.slice(0, 500)}`));

      // Parse output — stderr contains:
      //   ... debug lines ...
      //   <generated text>
      //   --- N tokens in Xms (Y tok/s) ---
      const lines = stderr.split('\n');

      // Find stats line
      let statsLine = '';
      let textEndIdx = lines.length;
      for (let i = lines.length - 1; i >= 0; i--) {
        if (lines[i].startsWith('---') && lines[i].includes('tok/s')) {
          statsLine = lines[i];
          textEndIdx = i;
          break;
        }
      }

      // Find where generated text starts (after "Output IDs: ..." line)
      let textStartIdx = 0;
      for (let i = 0; i < textEndIdx; i++) {
        if (lines[i].startsWith('Output IDs:')) {
          textStartIdx = i + 1;
          break;
        }
      }

      // Extract generated text, strip thinking tags and special tokens
      let text = lines.slice(textStartIdx, textEndIdx).join('\n').trim();
      text = text.replace(/<think>[\s\S]*?<\/think>\s*/g, '').trim();
      text = text.replace(/<\|im_end\|>/g, '').trim();

      // Parse stats
      const statsMatch = statsLine.match(/(\d+) tokens in ([\d.]+)ms \(([\d.]+) tok\/s\)/);
      const tokens = statsMatch ? parseInt(statsMatch[1], 10) : 0;
      const elapsed_ms = statsMatch ? parseFloat(statsMatch[2]) : 0;
      const tok_per_s = statsMatch ? parseFloat(statsMatch[3]) : 0;

      resolve({ text, tokens, elapsed_ms, tok_per_s });
    });
  });
}

// ---------------------------------------------------------------------------
// HTTP Server
// ---------------------------------------------------------------------------

const MIME = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
};

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks).toString()));
    req.on('error', reject);
  });
}

function json(res, status, data) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  });
  res.end(body);
}

const server = http.createServer(async (req, res) => {
  // CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    });
    return res.end();
  }

  const url = new URL(req.url, `http://${req.headers.host}`);

  // ---------- API routes ----------
  if (url.pathname === '/api/generate' && req.method === 'POST') {
    try {
      const body = JSON.parse(await readBody(req));
      const prompt = typeof body.prompt === 'string' ? body.prompt.slice(0, 4096) : 'Hello';
      const maxTokens = Math.min(Math.max(parseInt(body.max_tokens, 10) || cfg.maxTokens, 1), 2048);
      const result = await generate(prompt, maxTokens);
      return json(res, 200, result);
    } catch (err) {
      return json(res, 500, { error: err.message });
    }
  }

  if (url.pathname === '/api/health') {
    return json(res, 200, {
      status: 'ok',
      model: 'qwen3.5-0.8b',
      weights: path.basename(cfg.weights),
      binary: path.basename(cfg.binary),
    });
  }

  // ---------- Static files ----------
  let filePath = path.join(__dirname, url.pathname === '/' ? 'index.html' : url.pathname);
  filePath = path.normalize(filePath);

  // Prevent path traversal
  if (!filePath.startsWith(__dirname)) {
    res.writeHead(403);
    return res.end('Forbidden');
  }

  try {
    const data = fs.readFileSync(filePath);
    const ext = path.extname(filePath);
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('Not Found');
  }
});

server.listen(cfg.port, () => {
  console.log(`\n  Tomoul Qwen3.5-0.8B Chat Server`);
  console.log(`  ================================`);
  console.log(`  URL:        http://localhost:${cfg.port}`);
  console.log(`  Weights:    ${path.basename(cfg.weights)}`);
  console.log(`  Binary:     ${path.basename(cfg.binary)}`);
  console.log(`  Max tokens: ${cfg.maxTokens}\n`);
});
