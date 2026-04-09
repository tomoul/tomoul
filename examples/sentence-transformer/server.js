#!/usr/bin/env node
/**
 * Sentence Transformer Demo - Node.js Server
 *
 * HTTP server with REST API for generating sentence embeddings
 * using Tomoul's native sentence transformer library via FFI.
 *
 * Endpoints:
 *   POST /api/embed    - Embed a single text or array of texts
 *   GET  /api/status   - Check model readiness
 *   GET  /              - Interactive demo UI
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const koffi = require('koffi');

const PORT = process.env.PORT || 8003;
const EMBEDDING_DIM = 384;

// Determine platform
const platform = process.platform;
const libSuffix = platform === 'darwin' ? '.dylib' : '.so';

const PROJECT_ROOT = path.join(__dirname, '../..');
const LIB_PATH = path.join(PROJECT_ROOT, 'zig-out/lib', `libtomoul_sentence_transformer${libSuffix}`);
const LIB_PATH_ALT = path.join(PROJECT_ROOT, 'release/lib', `libtomoul_sentence_transformer${libSuffix}`);
const WEIGHTS_PATH = path.join(PROJECT_ROOT, 'artifacts/all_minilm_l6_v2.tl');
const VOCAB_PATH = path.join(PROJECT_ROOT, 'artifacts/all_minilm_l6_v2_vocab.txt');

const libPath = fs.existsSync(LIB_PATH) ? LIB_PATH : LIB_PATH_ALT;

if (!fs.existsSync(libPath)) {
    console.error(`Library not found. Build with: zig build lib -Dmodel=sentence_transformer -Doptimize=ReleaseFast`);
    process.exit(1);
}

console.log(`Loading library: ${libPath}`);

// Load FFI bindings
const lib = koffi.load(libPath);
const tomoul_sentence_transformer_init = lib.func('int tomoul_sentence_transformer_init(string, string)');
const tomoul_sentence_transformer_embed = lib.func('int tomoul_sentence_transformer_embed(_In_ uint8_t*, size_t, _Out_ float*)');
const tomoul_sentence_transformer_destroy = lib.func('void tomoul_sentence_transformer_destroy()');
const tomoul_sentence_transformer_is_ready = lib.func('int tomoul_sentence_transformer_is_ready()');

// Initialize model
console.log('Initializing model...');
console.log(`  Weights: ${WEIGHTS_PATH}`);
console.log(`  Vocab: ${VOCAB_PATH}`);

const initResult = tomoul_sentence_transformer_init(WEIGHTS_PATH, VOCAB_PATH);
if (initResult !== 0) {
    console.error(`Failed to initialize model, error code: ${initResult}`);
    process.exit(1);
}
console.log('Model initialized successfully!');

function embedText(text) {
    const inputBuffer = Buffer.from(text, 'utf8');
    const outputBuffer = Buffer.alloc(EMBEDDING_DIM * 4);
    const result = tomoul_sentence_transformer_embed(inputBuffer, inputBuffer.length, outputBuffer);
    if (result !== 0) {
        throw new Error(`Embedding failed: error ${result}`);
    }
    const embedding = new Array(EMBEDDING_DIM);
    for (let i = 0; i < EMBEDDING_DIM; i++) {
        embedding[i] = outputBuffer.readFloatLE(i * 4);
    }
    return embedding;
}

// MIME types
const MIME_TYPES = {
    '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
    '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml',
};

const server = http.createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
    }

    // POST /api/embed
    if (req.url === '/api/embed' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => { body += chunk.toString(); });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                const texts = Array.isArray(data.text) ? data.text : [data.text || ''];

                if (texts.length === 0 || (texts.length === 1 && !texts[0])) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'No text provided' }));
                    return;
                }

                const start = performance.now();
                const embeddings = texts.map(t => embedText(t));
                const elapsed = performance.now() - start;

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    embeddings,
                    dimensions: EMBEDDING_DIM,
                    count: embeddings.length,
                    elapsed_ms: parseFloat(elapsed.toFixed(2)),
                }));
            } catch (error) {
                console.error('Error:', error);
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: error.message }));
            }
        });
        return;
    }

    // GET /api/status
    if (req.url === '/api/status' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            ready: tomoul_sentence_transformer_is_ready() === 1,
            model: 'all-MiniLM-L6-v2',
            dimensions: EMBEDDING_DIM,
        }));
        return;
    }

    // Static file serving
    let filePath = req.url === '/' ? '/index.html' : req.url;
    filePath = path.join(__dirname, filePath);

    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';

    fs.readFile(filePath, (error, content) => {
        if (error) {
            if (error.code === 'ENOENT') {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Not found' }));
            } else {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Internal server error' }));
            }
        } else {
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

server.listen(PORT, () => {
    console.log(`\nSentence Transformer Demo running at http://localhost:${PORT}`);
    console.log(`  POST /api/embed   — generate embeddings`);
    console.log(`  GET  /api/status  — model status`);
});

process.on('SIGINT', () => {
    console.log('\nShutting down...');
    tomoul_sentence_transformer_destroy();
    process.exit(0);
});
