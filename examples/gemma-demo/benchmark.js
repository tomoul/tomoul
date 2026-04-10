#!/usr/bin/env node
/**
 * Gemma Benchmark - Node.js FFI (Tomoul)
 *
 * Loads the native Gemma library via koffi and benchmarks
 * text generation latency and throughput (tokens/sec).
 *
 * Compare against: python benchmark_pytorch.py
 */

const fs = require('fs');
const path = require('path');
const koffi = require('koffi');

// Determine platform
const platform = process.platform;
const libSuffix = platform === 'darwin' ? '.dylib' : '.so';

const PROJECT_ROOT = path.join(__dirname, '../..');

// Weight variant: 'f32' (default), 'q8', 'q8k', 'q4'
const VARIANT = process.env.TOMOUL_VARIANT || 'f32';
const weightFiles = {
    'f32': 'gemma_2b.tl',
    'q8': 'gemma_2b_q8.tl',
    'q8k': 'gemma_2b_q8k.tl',
    'q4': 'gemma_2b_q4.tl',
};

const LIB_PATH = path.join(PROJECT_ROOT, 'zig-out/lib', `libtomoul_gemma-2b${libSuffix}`);
const LIB_PATH_ALT = path.join(PROJECT_ROOT, 'release/lib', `libtomoul_gemma-2b${libSuffix}`);
const WEIGHTS_PATH = path.join(PROJECT_ROOT, 'artifacts', weightFiles[VARIANT] || weightFiles['f32']);
const TOKENIZER_PATH = path.join(PROJECT_ROOT, 'artifacts/gemma_vocab.bin');

const libPath = fs.existsSync(LIB_PATH) ? LIB_PATH : LIB_PATH_ALT;

if (!fs.existsSync(libPath)) {
    console.error(`Library not found at:\n  ${LIB_PATH}\n  ${LIB_PATH_ALT}`);
    console.error('\nBuild it with: zig build lib -Dmodel=gemma-2b -Doptimize=ReleaseFast');
    process.exit(1);
}

if (!fs.existsSync(WEIGHTS_PATH)) {
    console.error(`Weights not found: ${WEIGHTS_PATH}`);
    console.error('Export with: python3 tools/export_gemma.py --variant 2b -o artifacts/');
    process.exit(1);
}

if (!fs.existsSync(TOKENIZER_PATH)) {
    console.error(`Tokenizer not found: ${TOKENIZER_PATH}`);
    console.error('Export with: python3 tools/export_gemma.py --variant 2b -o artifacts/');
    process.exit(1);
}

const NUM_ITERATIONS = parseInt(process.env.BENCH_ITERS || '5', 10);
const MAX_TOKENS = parseInt(process.env.MAX_TOKENS || '64', 10);

// Test prompts of varying complexity
const PROMPTS = [
    "The meaning of life is",
    "In a world where technology",
    "Once upon a time in a land far away",
    "The quick brown fox",
    "Explain the theory of relativity in simple terms:",
];

console.log('============================================================');
console.log('Gemma 2B Benchmark (Tomoul via Node.js FFI)');
console.log(`  Model: Gemma 2B (${VARIANT.toUpperCase()})`);
console.log(`  Max tokens: ${MAX_TOKENS}`);
console.log('============================================================\n');

console.log(`Loading library: ${path.basename(libPath)}`);

// Load library
const lib = koffi.load(libPath);

// Define FFI bindings matching c.zig exports
const tomoul_gemma_init = lib.func('int tomoul_gemma_init(string, string)');
const tomoul_gemma_generate = lib.func('int tomoul_gemma_generate(const uint8_t*, size_t, int, float)');
const tomoul_gemma_get_output_buffer_ptr = lib.func('const uint8_t* tomoul_gemma_get_output_buffer_ptr()');
const tomoul_gemma_get_output_length = lib.func('size_t tomoul_gemma_get_output_length()');
const tomoul_gemma_destroy = lib.func('void tomoul_gemma_destroy()');
const tomoul_gemma_is_ready = lib.func('int tomoul_gemma_is_ready()');

// Initialize model
console.log('Loading model from disk...');
const loadStart = performance.now();
const initResult = tomoul_gemma_init(WEIGHTS_PATH, TOKENIZER_PATH);
const loadEnd = performance.now();

if (initResult !== 0) {
    console.error(`Failed to initialize model, error code: ${initResult}`);
    process.exit(1);
}

console.log(`Model loaded in ${((loadEnd - loadStart) / 1000).toFixed(2)}s`);
console.log(`Ready: ${tomoul_gemma_is_ready() === 1 ? 'yes' : 'no'}\n`);

// Generate function
function generateText(prompt, maxTokens, temperature) {
    const inputBuffer = Buffer.from(prompt, 'utf8');
    const result = tomoul_gemma_generate(inputBuffer, inputBuffer.length, maxTokens, temperature);
    if (result !== 0) {
        throw new Error(`Generation failed with error code: ${result}`);
    }

    const outputPtr = tomoul_gemma_get_output_buffer_ptr();
    const outputLen = tomoul_gemma_get_output_length();

    if (outputLen === 0) return '';

    const outputBuf = Buffer.from(koffi.decode(outputPtr, 'uint8_t', outputLen));
    return outputBuf.toString('utf8');
}

// --- Warm-up ---
console.log('Warm-up run...');
const warmupText = generateText(PROMPTS[0], 16, 0.0);
console.log(`  Prompt: "${PROMPTS[0]}"`);
console.log(`  Output: "${warmupText.slice(0, 100)}${warmupText.length > 100 ? '...' : ''}"`);

// --- Single prompt benchmark (greedy, temperature=0) ---
console.log(`\n--- Single Prompt (greedy, ${MAX_TOKENS} tokens max) ---`);
let times = [];
let tokenCounts = [];

for (let i = 0; i < NUM_ITERATIONS; i++) {
    const start = performance.now();
    const text = generateText(PROMPTS[0], MAX_TOKENS, 0.0);
    const end = performance.now();
    times.push(end - start);
    // Rough token count estimate: ~4 chars per token for English
    tokenCounts.push(Math.max(1, Math.round(text.length / 4)));
}

let avg = times.reduce((a, b) => a + b, 0) / times.length;
let sorted = [...times].sort((a, b) => a - b);
let p50 = sorted[Math.floor(sorted.length / 2)];
let minT = sorted[0];
let maxT = sorted[sorted.length - 1];
let avgTokens = tokenCounts.reduce((a, b) => a + b, 0) / tokenCounts.length;

console.log(`\n  Single prompt (${NUM_ITERATIONS} runs):`);
console.log(`    Average: ${avg.toFixed(1)}ms`);
console.log(`    Median:  ${p50.toFixed(1)}ms`);
console.log(`    Min:     ${minT.toFixed(1)}ms`);
console.log(`    Max:     ${maxT.toFixed(1)}ms`);
console.log(`    ~Tokens/sec: ${((avgTokens / (avg / 1000))).toFixed(1)}`);

const singleAvg = avg;
const singleP50 = p50;

// --- Multiple prompts (sequential, greedy) ---
console.log(`\n--- 5 Prompts (sequential, greedy, ${MAX_TOKENS} tokens each) ---`);
times = [];
for (let i = 0; i < NUM_ITERATIONS; i++) {
    const start = performance.now();
    for (const prompt of PROMPTS) {
        generateText(prompt, MAX_TOKENS, 0.0);
    }
    const end = performance.now();
    times.push(end - start);
}

avg = times.reduce((a, b) => a + b, 0) / times.length;
sorted = [...times].sort((a, b) => a - b);
p50 = sorted[Math.floor(sorted.length / 2)];
minT = sorted[0];

console.log(`  5 prompts sequential (${NUM_ITERATIONS} runs):`);
console.log(`    Average: ${avg.toFixed(1)}ms  (${(avg / 5).toFixed(1)}ms per prompt)`);
console.log(`    Median:  ${p50.toFixed(1)}ms  (${(p50 / 5).toFixed(1)}ms per prompt)`);
console.log(`    Min:     ${minT.toFixed(1)}ms  (${(minT / 5).toFixed(1)}ms per prompt)`);

const seq5Avg = avg;

// --- Short generation (16 tokens) for latency measurement ---
console.log(`\n--- Latency Test (16 tokens, greedy) ---`);
times = [];
for (let i = 0; i < NUM_ITERATIONS; i++) {
    const start = performance.now();
    generateText(PROMPTS[0], 16, 0.0);
    const end = performance.now();
    times.push(end - start);
}

avg = times.reduce((a, b) => a + b, 0) / times.length;
sorted = [...times].sort((a, b) => a - b);
p50 = sorted[Math.floor(sorted.length / 2)];
minT = sorted[0];

console.log(`  Short generation (${NUM_ITERATIONS} runs):`);
console.log(`    Average: ${avg.toFixed(1)}ms`);
console.log(`    Median:  ${p50.toFixed(1)}ms`);
console.log(`    Min:     ${minT.toFixed(1)}ms`);

const shortAvg = avg;
const shortP50 = p50;

// --- Sampling benchmark (temperature > 0) ---
console.log(`\n--- Sampling (temperature=0.7, ${MAX_TOKENS} tokens) ---`);
times = [];
const sampledOutputs = [];
for (let i = 0; i < NUM_ITERATIONS; i++) {
    const start = performance.now();
    const text = generateText(PROMPTS[0], MAX_TOKENS, 0.7);
    const end = performance.now();
    times.push(end - start);
    if (i < 3) sampledOutputs.push(text);
}

avg = times.reduce((a, b) => a + b, 0) / times.length;
sorted = [...times].sort((a, b) => a - b);
p50 = sorted[Math.floor(sorted.length / 2)];

console.log(`  Sampling (${NUM_ITERATIONS} runs):`);
console.log(`    Average: ${avg.toFixed(1)}ms`);
console.log(`    Median:  ${p50.toFixed(1)}ms`);

// Show diversity of sampled outputs
console.log('\n  Sample outputs (first 80 chars):');
for (let i = 0; i < sampledOutputs.length; i++) {
    const preview = sampledOutputs[i].slice(0, 80).replace(/\n/g, ' ');
    console.log(`    [${i}] "${preview}${sampledOutputs[i].length > 80 ? '...' : ''}"`);
}

const samplingAvg = avg;

// --- Correctness: deterministic greedy ---
console.log('\n--- Determinism Check (greedy should be identical) ---');
const outputs = [];
for (let i = 0; i < 3; i++) {
    outputs.push(generateText(PROMPTS[0], MAX_TOKENS, 0.0));
}
const allMatch = outputs.every(o => o === outputs[0]);
console.log(`  3 greedy runs match: ${allMatch ? 'PASS' : 'FAIL'}`);
if (!allMatch) {
    for (let i = 0; i < outputs.length; i++) {
        console.log(`    [${i}] "${outputs[i].slice(0, 60)}..."`);
    }
}

// Cleanup
tomoul_gemma_destroy();

// Summary
console.log('\n============================================================');
console.log('Summary (Tomoul / Node.js FFI)');
console.log('============================================================');
console.log(`  Variant:             ${VARIANT.toUpperCase()}`);
console.log(`  Max tokens:          ${MAX_TOKENS}`);
console.log(`  Single (greedy):     ${singleAvg.toFixed(1)}ms avg, ${singleP50.toFixed(1)}ms p50`);
console.log(`  Short (16 tok):      ${shortAvg.toFixed(1)}ms avg, ${shortP50.toFixed(1)}ms p50`);
console.log(`  5 seq (greedy):      ${seq5Avg.toFixed(1)}ms avg (${(seq5Avg / 5).toFixed(1)}ms/prompt)`);
console.log(`  Sampling (t=0.7):    ${samplingAvg.toFixed(1)}ms avg`);
console.log(`  Deterministic:       ${allMatch ? 'PASS' : 'FAIL'}`);
console.log('============================================================');
