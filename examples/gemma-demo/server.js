#!/usr/bin/env node
/**
 * Gemma Demo - Node.js Server
 *
 * HTTP server with REST API for text generation
 * using Tomoul's native Gemma library via FFI.
 *
 * Endpoints:
 *   POST /api/generate   - Generate text from a prompt
 *   GET  /api/status      - Check model readiness
 *   GET  /                - Interactive demo UI
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const koffi = require('koffi');

const PORT = process.env.PORT || 8010;

// Determine platform
const platform = process.platform;
const libSuffix = platform === 'darwin' ? '.dylib' : '.so';

const PROJECT_ROOT = path.join(__dirname, '../..');
const LIB_PATH = path.join(PROJECT_ROOT, 'zig-out/lib', `libtomoul_gemma-2b${libSuffix}`);
const LIB_PATH_ALT = path.join(PROJECT_ROOT, 'release/lib', `libtomoul_gemma-2b${libSuffix}`);
const WEIGHTS_PATH = path.join(PROJECT_ROOT, 'artifacts/gemma_2b.tl');
const TOKENIZER_PATH = path.join(PROJECT_ROOT, 'artifacts/gemma_vocab.bin');

const libPath = fs.existsSync(LIB_PATH) ? LIB_PATH : LIB_PATH_ALT;

if (!fs.existsSync(libPath)) {
    console.error(`Library not found. Build with: zig build lib -Dmodel=gemma-2b -Doptimize=ReleaseFast`);
    process.exit(1);
}

console.log(`Loading library: ${libPath}`);

// Load FFI bindings
const lib = koffi.load(libPath);
const tomoul_gemma_init = lib.func('int tomoul_gemma_init(string, string)');
const tomoul_gemma_generate = lib.func('int tomoul_gemma_generate(const uint8_t*, size_t, int, float)');
const tomoul_gemma_get_output_buffer_ptr = lib.func('const uint8_t* tomoul_gemma_get_output_buffer_ptr()');
const tomoul_gemma_get_output_length = lib.func('size_t tomoul_gemma_get_output_length()');
const tomoul_gemma_destroy = lib.func('void tomoul_gemma_destroy()');
const tomoul_gemma_is_ready = lib.func('int tomoul_gemma_is_ready()');

// Initialize model
console.log('Initializing model...');
console.log(`  Weights:   ${WEIGHTS_PATH}`);
console.log(`  Tokenizer: ${TOKENIZER_PATH}`);

const initResult = tomoul_gemma_init(WEIGHTS_PATH, TOKENIZER_PATH);
if (initResult !== 0) {
    console.error(`Failed to initialize model, error code: ${initResult}`);
    process.exit(1);
}
console.log('Model initialized successfully!');

function generateText(prompt, maxTokens, temperature) {
    const inputBuffer = Buffer.from(prompt, 'utf8');
    const result = tomoul_gemma_generate(inputBuffer, inputBuffer.length, maxTokens, temperature);
    if (result !== 0) {
        throw new Error(`Generation failed: error ${result}`);
    }

    const outputPtr = tomoul_gemma_get_output_buffer_ptr();
    const outputLen = tomoul_gemma_get_output_length();

    if (outputLen === 0) return '';

    const outputBuf = Buffer.from(koffi.decode(outputPtr, 'uint8_t', outputLen));
    return outputBuf.toString('utf8');
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

    // POST /api/generate
    if (req.url === '/api/generate' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => { body += chunk.toString(); });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                const prompt = data.prompt || '';
                const maxTokens = Math.min(Math.max(data.max_tokens || 128, 1), 2048);
                const temperature = Math.max(data.temperature || 0.0, 0.0);

                if (!prompt) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'No prompt provided' }));
                    return;
                }

                const start = performance.now();
                const text = generateText(prompt, maxTokens, temperature);
                const elapsed = performance.now() - start;

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    prompt,
                    generated: text,
                    max_tokens: maxTokens,
                    temperature,
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
            ready: tomoul_gemma_is_ready() === 1,
            model: 'gemma-2b',
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
    console.log(`\nGemma Demo running at http://localhost:${PORT}`);
    console.log(`  POST /api/generate  — generate text`);
    console.log(`  GET  /api/status    — model status`);
});

process.on('SIGINT', () => {
    console.log('\nShutting down...');
    tomoul_gemma_destroy();
    process.exit(0);
});
