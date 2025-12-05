#!/usr/bin/env node
/**
 * Punctuation Benchmark - Node.js
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
    libName = arch === 'x64'
        ? 'libtomoul_fullstop-punctuation-multilang-large_linux_x86_64.so'
        : 'libtomoul_fullstop-punctuation-multilang-large_linux_aarch64.so';
} else if (platform === 'darwin') {
    libName = arch === 'arm64'
        ? 'libtomoul_fullstop-punctuation-multilang-large_mac_aarch64.dylib'
        : 'libtomoul_fullstop-punctuation-multilang-large_mac_x86_64.dylib';
} else {
    console.error(`Unsupported platform: ${platform}`);
    process.exit(1);
}

const PROJECT_ROOT = path.join(__dirname, '../..');
const LIB_PATH = path.join(PROJECT_ROOT, 'release/lib', libName);
const WEIGHTS_PATH = path.join(PROJECT_ROOT, 'artifacts/fullstop_punctuation_multilang_large_q8.tl');
const VOCAB_PATH = path.join(PROJECT_ROOT, 'artifacts/fullstop_punctuation_multilang_large_vocab.txt');

// Check if library exists
if (!fs.existsSync(LIB_PATH)) {
    console.error(`Library not found: ${LIB_PATH}`);
    process.exit(1);
}

console.log('============================================================');
console.log('Tomoul Punctuation Benchmark (Node.js FFI)');
console.log('============================================================\n');

console.log(`Loading library: ${libName}`);

// Load library with koffi
const lib = koffi.load(LIB_PATH);

// Define FFI bindings
const tomoul_xlm_roberta_punctuation_init = lib.func('int tomoul_xlm_roberta_punctuation_init(string, string)');
const tomoul_xlm_roberta_punctuation_process = lib.func('int tomoul_xlm_roberta_punctuation_process(_In_ uint8_t*, size_t, _Out_ uint8_t*, size_t)');
const tomoul_xlm_roberta_punctuation_destroy = lib.func('void tomoul_xlm_roberta_punctuation_destroy()');
const tomoul_xlm_roberta_punctuation_version = lib.func('string tomoul_xlm_roberta_punctuation_version()');

// Initialize model (this includes loading from disk)
console.log('Loading model from disk...');
const loadStart = performance.now();
const initResult = tomoul_xlm_roberta_punctuation_init(WEIGHTS_PATH, VOCAB_PATH);
const loadEnd = performance.now();

if (initResult !== 0) {
    console.error(`Failed to initialize model, error code: ${initResult}`);
    process.exit(1);
}

console.log(`Model loaded in ${((loadEnd - loadStart) / 1000).toFixed(2)}s`);
console.log(`Version: ${tomoul_xlm_roberta_punctuation_version()}\n`);

// Process text function
function processText(inputText) {
    const inputBuffer = Buffer.from(inputText, 'utf8');
    const outputBuffer = Buffer.alloc(inputBuffer.length * 2);

    const resultLen = tomoul_xlm_roberta_punctuation_process(
        inputBuffer,
        inputBuffer.length,
        outputBuffer,
        outputBuffer.length
    );

    if (resultLen < 0) {
        throw new Error(`Processing failed with error code: ${resultLen}`);
    }

    return outputBuffer.toString('utf8', 0, resultLen);
}

// Test texts
const shortText = "hello world how are you doing today";
const longText = "hello world this is a test of the punctuation restoration system we are going to see how fast it can process this paragraph which contains exactly one hundred words or close to it the goal is to measure the performance difference between the python implementation using pytorch and the zig implementation using native code with manual memory management this benchmark will help us understand if the zero interpreter overhead and better memory control in zig provides a significant speedup compared to python we expect zig to be two to five times faster than python for this workload lets see if that prediction holds true";

const NUM_ITERATIONS = 10;

console.log('Running benchmark...\n');

// Warm-up run
console.log('Warm-up run...');
processText(shortText);

// Benchmark short text
console.log(`\n--- Short Text (${shortText.split(' ').length} words) ---`);
let times = [];
for (let i = 0; i < NUM_ITERATIONS; i++) {
    const start = performance.now();
    const result = processText(shortText);
    const end = performance.now();
    times.push(end - start);
    if (i === 0) {
        console.log(`Output: "${result}"`);
    }
}

let avg = times.reduce((a, b) => a + b, 0) / times.length;
let min = Math.min(...times);
let max = Math.max(...times);
console.log(`\nShort text (${NUM_ITERATIONS} runs):`);
console.log(`  Average: ${avg.toFixed(2)}ms`);
console.log(`  Min: ${min.toFixed(2)}ms`);
console.log(`  Max: ${max.toFixed(2)}ms`);

// Benchmark long text (100 words)
console.log(`\n--- Long Text (~100 words) ---`);
times = [];
for (let i = 0; i < NUM_ITERATIONS; i++) {
    const start = performance.now();
    const result = processText(longText);
    const end = performance.now();
    times.push(end - start);
    if (i === 0) {
        console.log(`Output: "${result.substring(0, 100)}..."`);
    }
}

avg = times.reduce((a, b) => a + b, 0) / times.length;
min = Math.min(...times);
max = Math.max(...times);
console.log(`\nLong text (${NUM_ITERATIONS} runs):`);
console.log(`  Average: ${avg.toFixed(2)}ms`);
console.log(`  Min: ${min.toFixed(2)}ms`);
console.log(`  Max: ${max.toFixed(2)}ms`);

// Cleanup
tomoul_xlm_roberta_punctuation_destroy();

console.log('\n============================================================');
console.log('Benchmark Complete!');
console.log('============================================================');
