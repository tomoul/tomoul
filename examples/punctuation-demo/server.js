#!/usr/bin/env node
/**
 * Punctuation Demo - Node.js Server
 *
 * Combined API + Static server for the punctuation restoration demo.
 * Uses FFI to call the native C library.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const koffi = require('koffi');

const PORT = process.env.PORT || 8001;
const API_PORT = process.env.API_PORT || 8002;

// Determine platform and architecture
const platform = process.platform;
const arch = process.arch;

let libName;
if (platform === 'linux') {
    if (arch === 'x64') {
        libName = 'libtomoul_fullstop-punctuation-multilang-large_linux_x86_64.so';
    } else if (arch === 'arm64') {
        libName = 'libtomoul_fullstop-punctuation-multilang-large_linux_aarch64.so';
    } else {
        console.error(`Unsupported Linux architecture: ${arch}`);
        process.exit(1);
    }
} else if (platform === 'darwin') {
    if (arch === 'arm64') {
        libName = 'libtomoul_fullstop-punctuation-multilang-large_mac_aarch64.dylib';
    } else if (arch === 'x64') {
        libName = 'libtomoul_fullstop-punctuation-multilang-large_mac_x86_64.dylib';
    } else {
        console.error(`Unsupported macOS architecture: ${arch}`);
        process.exit(1);
    }
} else {
    console.error(`Unsupported platform: ${platform}`);
    process.exit(1);
}

const PROJECT_ROOT = path.join(__dirname, '../..');
const LIB_PATH = path.join(PROJECT_ROOT, 'release/lib', libName);
const WEIGHTS_PATH = path.join(PROJECT_ROOT, 'artifacts/fullstop_punctuation_multilang_large.tl');
const VOCAB_PATH = path.join(PROJECT_ROOT, 'artifacts/fullstop_punctuation_multilang_large_vocab.txt');

// Check if library exists
if (!fs.existsSync(LIB_PATH)) {
    console.error(`Library not found: ${LIB_PATH}`);
    console.error('Please build the libraries first: zig build lib -Dmodel=fullstop-punctuation-multilang-large');
    process.exit(1);
}

console.log(`Loading library: ${LIB_PATH}`);

// Load library with koffi
const lib = koffi.load(LIB_PATH);

// Define FFI bindings
const tomoul_xlm_roberta_punctuation_init = lib.func('int tomoul_xlm_roberta_punctuation_init(string, string)');
const tomoul_xlm_roberta_punctuation_process = lib.func('int tomoul_xlm_roberta_punctuation_process(_In_ uint8_t*, size_t, _Out_ uint8_t*, size_t)');
const tomoul_xlm_roberta_punctuation_is_ready = lib.func('int tomoul_xlm_roberta_punctuation_is_ready()');
const tomoul_xlm_roberta_punctuation_destroy = lib.func('void tomoul_xlm_roberta_punctuation_destroy()');
const tomoul_xlm_roberta_punctuation_version = lib.func('string tomoul_xlm_roberta_punctuation_version()');

// Initialize model
console.log('Initializing model...');
console.log(`  Weights: ${WEIGHTS_PATH}`);
console.log(`  Vocab: ${VOCAB_PATH}`);

if (!fs.existsSync(WEIGHTS_PATH)) {
    console.error(`Weights file not found: ${WEIGHTS_PATH}`);
    process.exit(1);
}

if (!fs.existsSync(VOCAB_PATH)) {
    console.error(`Vocab file not found: ${VOCAB_PATH}`);
    process.exit(1);
}

const initResult = tomoul_xlm_roberta_punctuation_init(WEIGHTS_PATH, VOCAB_PATH);
if (initResult !== 0) {
    console.error(`Failed to initialize model, error code: ${initResult}`);
    process.exit(1);
}

const version = tomoul_xlm_roberta_punctuation_version();
console.log(`Model initialized successfully!`);
console.log(`Model version: ${version}`);

// Process text function
function processText(inputText) {
    const inputBuffer = Buffer.from(inputText, 'utf8');
    const outputBuffer = Buffer.alloc(inputBuffer.length * 2); // Generous buffer

    const resultLen = tomoul_xlm_roberta_punctuation_process(
        inputBuffer,
        inputBuffer.length,
        outputBuffer,
        outputBuffer.length
    );

    if (resultLen < 0) {
        const errorCodes = {
            '-1': 'Model not initialized',
            '-2': 'Invalid input length',
            '-3': 'Processing failed',
            '-4': 'Output buffer too small'
        };
        throw new Error(errorCodes[resultLen] || `Unknown error code: ${resultLen}`);
    }

    return outputBuffer.toString('utf8', 0, resultLen);
}

// MIME types
const MIME_TYPES = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon'
};

// Create HTTP server
const server = http.createServer((req, res) => {
    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    // Handle OPTIONS
    if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
    }

    // API endpoints
    if (req.url === '/api/punctuate' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                const inputText = data.text || '';

                if (!inputText) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'No text provided' }));
                    return;
                }

                const output = processText(inputText);

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    input: inputText,
                    output: output,
                    version: version
                }));
            } catch (error) {
                console.error('Processing error:', error);
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: error.message }));
            }
        });
        return;
    }

    if (req.url === '/api/status' && req.method === 'GET') {
        const isReady = tomoul_xlm_roberta_punctuation_is_ready();
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            ready: Boolean(isReady),
            version: version,
            model: 'fullstop-punctuation-multilang-large'
        }));
        return;
    }

    // Static file serving
    let filePath = req.url === '/' ? '/index.html' : req.url;
    filePath = path.join(__dirname, filePath);

    const extname = String(path.extname(filePath)).toLowerCase();
    const contentType = MIME_TYPES[extname] || 'application/octet-stream';

    fs.readFile(filePath, (error, content) => {
        if (error) {
            if (error.code === 'ENOENT') {
                res.writeHead(404);
                res.end('404 Not Found');
            } else {
                res.writeHead(500);
                res.end(`Server Error: ${error.code}`);
            }
        } else {
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

// Cleanup on exit
process.on('SIGINT', () => {
    console.log('\nShutting down...');
    tomoul_xlm_roberta_punctuation_destroy();
    process.exit(0);
});

// Start server
server.listen(PORT, () => {
    console.log(`
╔══════════════════════════════════════════════════════════════╗
║        Tomoul Punctuation Demo Server (Node.js)              ║
╠══════════════════════════════════════════════════════════════╣
║  Combined API + Web Server                                   ║
║    - Demo UI → http://localhost:${PORT}/                     ║
║    - API     → http://localhost:${PORT}/api/punctuate        ║
║    - Status  → http://localhost:${PORT}/api/status           ║
║                                                              ║
║  Open http://localhost:${PORT}/ in your browser              ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
`);
});
