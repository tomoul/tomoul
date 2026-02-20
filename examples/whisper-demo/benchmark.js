#!/usr/bin/env node
/**
 * Whisper Benchmark - Node.js
 *
 * Loads the model once, then runs multiple inference calls to get accurate timing.
 */

const fs = require('fs');
const path = require('path');
const koffi = require('koffi');

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

console.log('============================================================');
console.log('Tomoul Whisper Benchmark (Node.js FFI)');
console.log('============================================================\n');

console.log(`Loading library: ${libName}`);

// Load library with koffi
const lib = koffi.load(LIB_PATH);

// Define FFI bindings
const tomoul_whisper_init = lib.func('int tomoul_whisper_init(string)');
const tomoul_whisper_transcribe_file = lib.func('int tomoul_whisper_transcribe_file(string, _Out_ uint32_t*, size_t)');
const tomoul_whisper_destroy = lib.func('void tomoul_whisper_destroy()');
const tomoul_whisper_version = lib.func('string tomoul_whisper_version()');
const tomoul_whisper_variant = lib.func('string tomoul_whisper_variant()');

// Check weights file
if (!fs.existsSync(WEIGHTS_PATH)) {
    console.error(`Weights file not found: ${WEIGHTS_PATH}`);
    console.error('Please ensure whisper_tiny.tl exists in the artifacts directory.');
    process.exit(1);
}

// Check audio file
if (!fs.existsSync(TEST_AUDIO_PATH)) {
    console.error(`Test audio file not found: ${TEST_AUDIO_PATH}`);
    console.error('Please ensure english_man.wav exists in the models directory.');
    process.exit(1);
}

// Initialize model (this includes loading from disk)
console.log('Loading model from disk...');
const loadStart = performance.now();
const initResult = tomoul_whisper_init(WEIGHTS_PATH);
const loadEnd = performance.now();

if (initResult !== 0) {
    console.error(`Failed to initialize model, error code: ${initResult}`);
    process.exit(1);
}

console.log(`Model loaded in ${((loadEnd - loadStart) / 1000).toFixed(2)}s`);
console.log(`Version: ${tomoul_whisper_version()}`);
console.log(`Variant: ${tomoul_whisper_variant()}\n`);

// Transcribe function
function transcribeFile(audioPath) {
    const outputBuffer = Buffer.alloc(1024 * 4); // Up to 1024 tokens

    const numTokens = tomoul_whisper_transcribe_file(
        audioPath,
        outputBuffer,
        1024
    );

    if (numTokens < 0) {
        throw new Error(`Transcription failed with error code: ${numTokens}`);
    }

    // Read token IDs from buffer
    const tokens = [];
    for (let i = 0; i < numTokens; i++) {
        tokens.push(outputBuffer.readUInt32LE(i * 4));
    }

    return tokens;
}

const NUM_ITERATIONS = 5;

console.log('Running benchmark...\n');
console.log(`Test audio: ${TEST_AUDIO_PATH}`);

// Warm-up run
console.log('Warm-up run...');
const warmupTokens = transcribeFile(TEST_AUDIO_PATH);
console.log(`  Warm-up tokens: ${warmupTokens.length}`);

// Benchmark
console.log(`\n--- Transcription (${NUM_ITERATIONS} runs) ---`);
let times = [];
let firstTokens = null;

for (let i = 0; i < NUM_ITERATIONS; i++) {
    const start = performance.now();
    const tokens = transcribeFile(TEST_AUDIO_PATH);
    const end = performance.now();
    times.push(end - start);

    if (i === 0) {
        firstTokens = tokens;
        console.log(`  Run ${i + 1}: ${(end - start).toFixed(2)}ms - ${tokens.length} tokens`);
        console.log(`  Token IDs: [${tokens.slice(0, 10).join(', ')}${tokens.length > 10 ? '...' : ''}]`);
    } else {
        console.log(`  Run ${i + 1}: ${(end - start).toFixed(2)}ms - ${tokens.length} tokens`);
    }
}

const avg = times.reduce((a, b) => a + b, 0) / times.length;
const min = Math.min(...times);
const max = Math.max(...times);

console.log(`\nResults (${NUM_ITERATIONS} runs):`);
console.log(`  Average: ${avg.toFixed(2)}ms`);
console.log(`  Min: ${min.toFixed(2)}ms`);
console.log(`  Max: ${max.toFixed(2)}ms`);
console.log(`  Tokens: ${firstTokens.length}`);

// Cleanup
tomoul_whisper_destroy();

console.log('\n============================================================');
console.log('Benchmark Complete!');
console.log('============================================================');
