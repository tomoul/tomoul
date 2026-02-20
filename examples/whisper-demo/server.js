#!/usr/bin/env node
/**
 * Whisper Demo - Node.js Server
 *
 * Combined API + Static server for the Whisper speech-to-text demo.
 * Uses FFI to call the native C library.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const koffi = require('koffi');

const PORT = process.env.PORT || 8003;

// Determine platform and architecture
const platform = process.platform;
const arch = process.arch;

let libName;
if (platform === 'linux') {
    if (arch === 'x64') {
        libName = 'libtomoul_whisper-tiny_linux_x86_64.so';
    } else if (arch === 'arm64') {
        libName = 'libtomoul_whisper-tiny_linux_aarch64.so';
    } else {
        console.error(`Unsupported Linux architecture: ${arch}`);
        process.exit(1);
    }
} else if (platform === 'darwin') {
    if (arch === 'arm64') {
        libName = 'libtomoul_whisper-tiny_mac_aarch64.dylib';
    } else if (arch === 'x64') {
        libName = 'libtomoul_whisper-tiny_mac_x86_64.dylib';
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
const WEIGHTS_PATH = path.join(PROJECT_ROOT, 'models/whisper_tiny.tl');
const TEST_AUDIO_PATH = path.join(PROJECT_ROOT, 'models/english_man.wav');

// Check if library exists
if (!fs.existsSync(LIB_PATH)) {
    console.error(`Library not found: ${LIB_PATH}`);
    console.error('Please build the libraries first: python3 scripts/release.py --model whisper-tiny --build-only');
    process.exit(1);
}

console.log(`Loading library: ${LIB_PATH}`);

// Load library with koffi
const lib = koffi.load(LIB_PATH);

// Define FFI bindings
const tomoul_whisper_init = lib.func('int tomoul_whisper_init(string)');
const tomoul_whisper_transcribe_file = lib.func('int tomoul_whisper_transcribe_file(string, _Out_ uint32_t*, size_t)');
const tomoul_whisper_is_ready = lib.func('int tomoul_whisper_is_ready()');
const tomoul_whisper_destroy = lib.func('void tomoul_whisper_destroy()');
const tomoul_whisper_version = lib.func('string tomoul_whisper_version()');
const tomoul_whisper_variant = lib.func('string tomoul_whisper_variant()');

// Initialize model
console.log('Initializing model...');
console.log(`  Weights: ${WEIGHTS_PATH}`);

if (!fs.existsSync(WEIGHTS_PATH)) {
    console.error(`Weights file not found: ${WEIGHTS_PATH}`);
    console.error('Please download or generate the whisper_tiny.tl weights file first.');
    process.exit(1);
}

const initResult = tomoul_whisper_init(WEIGHTS_PATH);
if (initResult !== 0) {
    console.error(`Failed to initialize model, error code: ${initResult}`);
    process.exit(1);
}

const version = tomoul_whisper_version();
const variant = tomoul_whisper_variant();
console.log(`Model initialized successfully!`);
console.log(`Model version: ${version}`);
console.log(`Model variant: ${variant}`);

// Whisper tokenizer (simplified - we just return raw token IDs for now)
// Full BPE decoding would require the tiktoken or similar tokenizer

// Transcribe audio file function
function transcribeFile(audioPath) {
    const outputBuffer = Buffer.alloc(1024 * 4); // Up to 1024 tokens * 4 bytes each

    const numTokens = tomoul_whisper_transcribe_file(
        audioPath,
        outputBuffer,
        1024
    );

    if (numTokens < 0) {
        const errorCodes = {
            '-1': 'Model not initialized',
            '-2': 'Failed to load audio file',
            '-3': 'Failed to compute mel spectrogram',
            '-4': 'Transcription failed',
            '-5': 'Output buffer too small'
        };
        throw new Error(errorCodes[numTokens] || `Unknown error code: ${numTokens}`);
    }

    // Read token IDs from buffer
    const tokens = [];
    for (let i = 0; i < numTokens; i++) {
        tokens.push(outputBuffer.readUInt32LE(i * 4));
    }

    return tokens;
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
    '.ico': 'image/x-icon',
    '.wav': 'audio/wav'
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
    if (req.url === '/api/transcribe' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                const audioPath = data.audio_path || TEST_AUDIO_PATH;

                if (!fs.existsSync(audioPath)) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: `Audio file not found: ${audioPath}` }));
                    return;
                }

                const startTime = performance.now();
                const tokens = transcribeFile(audioPath);
                const endTime = performance.now();

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    audio_path: audioPath,
                    tokens: tokens,
                    num_tokens: tokens.length,
                    inference_time_ms: (endTime - startTime).toFixed(2),
                    version: version,
                    variant: variant
                }));
            } catch (error) {
                console.error('Transcription error:', error);
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: error.message }));
            }
        });
        return;
    }

    if (req.url === '/api/status' && req.method === 'GET') {
        const isReady = tomoul_whisper_is_ready();
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            ready: Boolean(isReady),
            version: version,
            variant: variant,
            model: 'whisper-tiny'
        }));
        return;
    }

    if (req.url === '/api/test' && req.method === 'GET') {
        // Quick test endpoint using the test audio file
        try {
            if (!fs.existsSync(TEST_AUDIO_PATH)) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: `Test audio file not found: ${TEST_AUDIO_PATH}` }));
                return;
            }

            const startTime = performance.now();
            const tokens = transcribeFile(TEST_AUDIO_PATH);
            const endTime = performance.now();

            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                audio_path: TEST_AUDIO_PATH,
                tokens: tokens,
                num_tokens: tokens.length,
                inference_time_ms: (endTime - startTime).toFixed(2),
                version: version,
                variant: variant
            }));
        } catch (error) {
            console.error('Test error:', error);
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: error.message }));
        }
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
    tomoul_whisper_destroy();
    process.exit(0);
});

// Start server
server.listen(PORT, () => {
    console.log(`
======================================================================
        Tomoul Whisper Demo Server (Node.js)
======================================================================
  Combined API + Web Server
    - Demo UI  -> http://localhost:${PORT}/
    - API      -> http://localhost:${PORT}/api/transcribe
    - Status   -> http://localhost:${PORT}/api/status
    - Test     -> http://localhost:${PORT}/api/test

  Test audio file: ${TEST_AUDIO_PATH}

  Open http://localhost:${PORT}/ in your browser
  Press Ctrl+C to stop
======================================================================
`);
});
