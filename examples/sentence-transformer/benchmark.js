#!/usr/bin/env node
/**
 * Sentence Transformer Benchmark - Node.js FFI (Tomoul)
 *
 * Loads the native sentence transformer library via koffi and benchmarks
 * embedding generation for single and batch sentences.
 *
 * Compare against: python benchmark_pytorch.py
 */

const fs = require('fs');
const path = require('path');
const koffi = require('koffi');

// Determine platform and architecture
const platform = process.platform;
const arch = process.arch;

let libSuffix;
if (platform === 'linux') {
    libSuffix = arch === 'x64' ? '.so' : '.so';
} else if (platform === 'darwin') {
    libSuffix = '.dylib';
} else {
    console.error(`Unsupported platform: ${platform}`);
    process.exit(1);
}

const PROJECT_ROOT = path.join(__dirname, '../..');

// Weight variant: 'f32' (default), 'q8k', or 'f16'
const VARIANT = process.env.TOMOUL_VARIANT || 'f32';
const weightFiles = {
    'f32': 'all_minilm_l6_v2.tl',
    'q8k': 'all_minilm_l6_v2_q8k.tl',
    'f16': 'all_minilm_l6_v2_f16.tl',
};

const LIB_PATH = path.join(PROJECT_ROOT, 'zig-out/lib', `libtomoul_sentence_transformer${libSuffix}`);
const WEIGHTS_PATH = path.join(PROJECT_ROOT, 'artifacts', weightFiles[VARIANT] || weightFiles['f32']);
const VOCAB_PATH = path.join(PROJECT_ROOT, 'artifacts/all_minilm_l6_v2_vocab.txt');

// Also check release/lib/ if zig-out doesn't have it
const LIB_PATH_ALT = path.join(PROJECT_ROOT, 'release/lib', `libtomoul_sentence_transformer${libSuffix}`);

const libPath = fs.existsSync(LIB_PATH) ? LIB_PATH : LIB_PATH_ALT;

if (!fs.existsSync(libPath)) {
    console.error(`Library not found at:\n  ${LIB_PATH}\n  ${LIB_PATH_ALT}`);
    console.error('\nBuild it with: zig build lib -Dmodel=sentence_transformer -Doptimize=ReleaseFast');
    process.exit(1);
}

if (!fs.existsSync(WEIGHTS_PATH)) {
    console.error(`Weights not found: ${WEIGHTS_PATH}`);
    console.error('Export with: python3 tools/export_sentence_transformer.py -o artifacts/');
    process.exit(1);
}

if (!fs.existsSync(VOCAB_PATH)) {
    console.error(`Vocab not found: ${VOCAB_PATH}`);
    process.exit(1);
}

const EMBEDDING_DIM = 384;
const NUM_ITERATIONS = parseInt(process.env.BENCH_ITERS || '20', 10);

// Test sentences (same as Python benchmark)
const SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "I had pizza for lunch yesterday",
    "The capital of France is Paris",
    "Quantum computing uses qubits instead of classical bits",
];

const BATCH_SENTENCES = [
    "The weather is nice today",
    "Python is a popular programming language",
    "Neural networks can learn complex patterns",
    "The stock market closed higher on Friday",
    "Renewable energy is becoming more affordable",
    "The quick brown fox jumps over the lazy dog",
    "Deep learning has revolutionized computer vision",
    "Coffee is one of the most consumed beverages worldwide",
    "The Olympic Games are held every four years",
    "Artificial intelligence is transforming healthcare",
];

console.log('============================================================');
console.log('Sentence Transformer Benchmark (Tomoul via Node.js FFI)');
console.log('  Model: all-MiniLM-L6-v2 (384-dim)');
console.log('============================================================\n');

console.log(`Loading library: ${path.basename(libPath)}`);

// Load library
const lib = koffi.load(libPath);

// Define FFI bindings matching c.zig exports
const tomoul_sentence_transformer_init = lib.func('int tomoul_sentence_transformer_init(string, string)');
const tomoul_sentence_transformer_embed = lib.func('int tomoul_sentence_transformer_embed(_In_ uint8_t*, size_t, _Out_ float*)');
const tomoul_sentence_transformer_embed_batch = lib.func('int tomoul_sentence_transformer_embed_batch(_In_ const uint8_t**, _In_ const size_t*, size_t, _Out_ float*)');
const tomoul_sentence_transformer_destroy = lib.func('void tomoul_sentence_transformer_destroy()');
const tomoul_sentence_transformer_is_ready = lib.func('int tomoul_sentence_transformer_is_ready()');

// GPU FFI bindings
const tomoul_sentence_transformer_gpu_init = lib.func('int tomoul_sentence_transformer_gpu_init()');
const tomoul_sentence_transformer_gpu_is_active = lib.func('int tomoul_sentence_transformer_gpu_is_active()');
const tomoul_sentence_transformer_gpu_device_name = lib.func('string tomoul_sentence_transformer_gpu_device_name()');
const tomoul_sentence_transformer_gpu_embed = lib.func('int tomoul_sentence_transformer_gpu_embed(_In_ uint8_t*, size_t, _Out_ float*)');
const tomoul_sentence_transformer_gpu_embed_batch = lib.func('int tomoul_sentence_transformer_gpu_embed_batch(_In_ const uint8_t**, _In_ const size_t*, size_t, _Out_ float*)');
const tomoul_sentence_transformer_gpu_destroy = lib.func('void tomoul_sentence_transformer_gpu_destroy()');

// Initialize model
console.log('Loading model from disk...');
const loadStart = performance.now();
const initResult = tomoul_sentence_transformer_init(WEIGHTS_PATH, VOCAB_PATH);
const loadEnd = performance.now();

if (initResult !== 0) {
    console.error(`Failed to initialize model, error code: ${initResult}`);
    process.exit(1);
}

console.log(`Model loaded in ${((loadEnd - loadStart) / 1000).toFixed(2)}s`);
console.log(`Ready: ${tomoul_sentence_transformer_is_ready() === 1 ? 'yes' : 'no'}\n`);

// Embed function
function embedText(text) {
    const inputBuffer = Buffer.from(text, 'utf8');
    const outputBuffer = Buffer.alloc(EMBEDDING_DIM * 4); // f32 = 4 bytes

    const result = tomoul_sentence_transformer_embed(inputBuffer, inputBuffer.length, outputBuffer);
    if (result !== 0) {
        throw new Error(`Embedding failed with error code: ${result}`);
    }

    // Parse f32 array from buffer
    const embedding = new Float32Array(EMBEDDING_DIM);
    for (let i = 0; i < EMBEDDING_DIM; i++) {
        embedding[i] = outputBuffer.readFloatLE(i * 4);
    }
    return embedding;
}

// Batch embed function
function embedBatchTexts(texts) {
    const batchSize = texts.length;
    const textBuffers = texts.map(t => Buffer.from(t, 'utf8'));
    const textLengths = textBuffers.map(b => b.length);
    const outputBuffer = Buffer.alloc(batchSize * EMBEDDING_DIM * 4);

    const result = tomoul_sentence_transformer_embed_batch(textBuffers, textLengths, batchSize, outputBuffer);
    if (result !== 0) {
        throw new Error(`Batch embedding failed with error code: ${result}`);
    }

    const embeddings = [];
    for (let i = 0; i < batchSize; i++) {
        const emb = new Float32Array(EMBEDDING_DIM);
        for (let j = 0; j < EMBEDDING_DIM; j++) {
            emb[j] = outputBuffer.readFloatLE((i * EMBEDDING_DIM + j) * 4);
        }
        embeddings.push(emb);
    }
    return embeddings;
}

function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function l2Norm(v) {
    let sum = 0;
    for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
    return Math.sqrt(sum);
}

// --- Warm-up ---
console.log('Warm-up run...');
const warmupEmb = embedText(SENTENCES[0]);
console.log(`  Embedding dim: ${warmupEmb.length}`);
console.log(`  L2 norm: ${l2Norm(warmupEmb).toFixed(6)}`);
console.log(`  First 5 values: [${Array.from(warmupEmb.slice(0, 5)).map(v => v.toFixed(6)).join(', ')}]`);

// --- Single sentence benchmark ---
console.log(`\n--- Single Sentence ---`);
let times = [];
for (let i = 0; i < NUM_ITERATIONS; i++) {
    const start = performance.now();
    embedText(SENTENCES[0]);
    const end = performance.now();
    times.push(end - start);
}

let avg = times.reduce((a, b) => a + b, 0) / times.length;
let sorted = [...times].sort((a, b) => a - b);
let p50 = sorted[Math.floor(sorted.length / 2)];
let min = sorted[0];
let max = sorted[sorted.length - 1];

console.log(`\n  Single sentence (${NUM_ITERATIONS} runs):`);
console.log(`    Average: ${avg.toFixed(2)}ms`);
console.log(`    Median:  ${p50.toFixed(2)}ms`);
console.log(`    Min:     ${min.toFixed(2)}ms`);
console.log(`    Max:     ${max.toFixed(2)}ms`);

const singleAvg = avg;
const singleP50 = p50;

// --- 5 sentences sequential ---
console.log(`\n--- 5 Sentences (sequential) ---`);
times = [];
for (let i = 0; i < NUM_ITERATIONS; i++) {
    const start = performance.now();
    for (const sentence of SENTENCES) {
        embedText(sentence);
    }
    const end = performance.now();
    times.push(end - start);
}

avg = times.reduce((a, b) => a + b, 0) / times.length;
sorted = [...times].sort((a, b) => a - b);
p50 = sorted[Math.floor(sorted.length / 2)];
min = sorted[0];

console.log(`  5 sentences sequential (${NUM_ITERATIONS} runs):`);
console.log(`    Average: ${avg.toFixed(2)}ms  (${(avg / 5).toFixed(2)}ms per sentence)`);
console.log(`    Median:  ${p50.toFixed(2)}ms  (${(p50 / 5).toFixed(2)}ms per sentence)`);
console.log(`    Min:     ${min.toFixed(2)}ms  (${(min / 5).toFixed(2)}ms per sentence)`);

const seq5Avg = avg;

// --- 10 sentences sequential (no native batch API yet) ---
console.log(`\n--- 10 Sentences (sequential) ---`);
times = [];
for (let i = 0; i < NUM_ITERATIONS; i++) {
    const start = performance.now();
    for (const sentence of BATCH_SENTENCES) {
        embedText(sentence);
    }
    const end = performance.now();
    times.push(end - start);
}

avg = times.reduce((a, b) => a + b, 0) / times.length;
sorted = [...times].sort((a, b) => a - b);
p50 = sorted[Math.floor(sorted.length / 2)];
min = sorted[0];

console.log(`  10 sentences sequential (${NUM_ITERATIONS} runs):`);
console.log(`    Average: ${avg.toFixed(2)}ms  (${(avg / 10).toFixed(2)}ms per sentence)`);
console.log(`    Median:  ${p50.toFixed(2)}ms  (${(p50 / 10).toFixed(2)}ms per sentence)`);
console.log(`    Min:     ${min.toFixed(2)}ms  (${(min / 10).toFixed(2)}ms per sentence)`);

const batch10Avg = avg;

// --- 5 sentences BATCHED ---
console.log(`\n--- 5 Sentences (batched) ---`);
times = [];
const batch5Texts = BATCH_SENTENCES.slice(0, 5);
for (let i = 0; i < NUM_ITERATIONS; i++) {
    const start = performance.now();
    embedBatchTexts(batch5Texts);
    const end = performance.now();
    times.push(end - start);
}

avg = times.reduce((a, b) => a + b, 0) / times.length;
sorted = [...times].sort((a, b) => a - b);
p50 = sorted[Math.floor(sorted.length / 2)];
min = sorted[0];

console.log(`  5 sentences batched (${NUM_ITERATIONS} runs):`);
console.log(`    Average: ${avg.toFixed(2)}ms  (${(avg / 5).toFixed(2)}ms per sentence)`);
console.log(`    Median:  ${p50.toFixed(2)}ms  (${(p50 / 5).toFixed(2)}ms per sentence)`);
console.log(`    Min:     ${min.toFixed(2)}ms  (${(min / 5).toFixed(2)}ms per sentence)`);
console.log(`    Speedup vs sequential: ${(seq5Avg / avg).toFixed(2)}x`);

const batched5Avg = avg;

// --- 10 sentences BATCHED ---
console.log(`\n--- 10 Sentences (batched) ---`);
times = [];
for (let i = 0; i < NUM_ITERATIONS; i++) {
    const start = performance.now();
    embedBatchTexts(BATCH_SENTENCES);
    const end = performance.now();
    times.push(end - start);
}

avg = times.reduce((a, b) => a + b, 0) / times.length;
sorted = [...times].sort((a, b) => a - b);
p50 = sorted[Math.floor(sorted.length / 2)];
min = sorted[0];

console.log(`  10 sentences batched (${NUM_ITERATIONS} runs):`);
console.log(`    Average: ${avg.toFixed(2)}ms  (${(avg / 10).toFixed(2)}ms per sentence)`);
console.log(`    Median:  ${p50.toFixed(2)}ms  (${(p50 / 10).toFixed(2)}ms per sentence)`);
console.log(`    Min:     ${min.toFixed(2)}ms  (${(min / 10).toFixed(2)}ms per sentence)`);
console.log(`    Speedup vs sequential: ${(batch10Avg / avg).toFixed(2)}x`);

const batched10Avg = avg;

// --- Correctness: cosine similarities between sentence pairs ---
console.log('\n--- Correctness Check ---');
const embeddings = SENTENCES.map(s => embedText(s));
for (let i = 0; i < SENTENCES.length; i++) {
    for (let j = i + 1; j < SENTENCES.length; j++) {
        const sim = cosineSimilarity(embeddings[i], embeddings[j]);
        console.log(`  [${i}] vs [${j}]: ${sim.toFixed(4)}`);
    }
}

// --- Batch vs Sequential correctness ---
console.log('\n--- Batch vs Sequential Correctness ---');
const batchEmbs = embedBatchTexts(SENTENCES);
let maxDiff = 0;
for (let i = 0; i < SENTENCES.length; i++) {
    const sim = cosineSimilarity(embeddings[i], batchEmbs[i]);
    let diff = 0;
    for (let j = 0; j < EMBEDDING_DIM; j++) {
        diff = Math.max(diff, Math.abs(embeddings[i][j] - batchEmbs[i][j]));
    }
    maxDiff = Math.max(maxDiff, diff);
    console.log(`  Sentence ${i}: cosine=${sim.toFixed(6)}, max_abs_diff=${diff.toExponential(3)}`);
}
console.log(`  Overall max abs diff: ${maxDiff.toExponential(3)}`);
console.log(`  Match: ${maxDiff < 1e-4 ? 'PASS' : 'FAIL (divergence > 1e-4)'}`);

// =============================================================================
// GPU Benchmark (if available)
// =============================================================================

let gpuSingleAvg = 0;
let gpuBatch5Avg = 0;
let gpuBatch10Avg = 0;
let gpuAvailable = false;

const USE_GPU = process.env.TOMOUL_GPU !== '0'; // disable with TOMOUL_GPU=0

if (USE_GPU) {
    console.log('\n============================================================');
    console.log('GPU Acceleration');
    console.log('============================================================\n');

    console.log('Initializing GPU backend...');
    const gpuInitStart = performance.now();
    const gpuResult = tomoul_sentence_transformer_gpu_init();
    const gpuInitEnd = performance.now();

    if (gpuResult !== 0) {
        console.log(`  GPU init failed (code: ${gpuResult}) — skipping GPU benchmarks`);
    } else {
        gpuAvailable = tomoul_sentence_transformer_gpu_is_active() === 1;
        const deviceName = tomoul_sentence_transformer_gpu_device_name();
        console.log(`  GPU initialized in ${((gpuInitEnd - gpuInitStart) / 1000).toFixed(2)}s`);
        console.log(`  Device: ${deviceName}`);
        console.log(`  Active: ${gpuAvailable ? 'yes' : 'no (CPU fallback)'}`);

        // GPU embed functions
        function gpuEmbedText(text) {
            const inputBuffer = Buffer.from(text, 'utf8');
            const outputBuffer = Buffer.alloc(EMBEDDING_DIM * 4);

            const result = tomoul_sentence_transformer_gpu_embed(inputBuffer, inputBuffer.length, outputBuffer);
            if (result !== 0) {
                throw new Error(`GPU embedding failed with error code: ${result}`);
            }

            const embedding = new Float32Array(EMBEDDING_DIM);
            for (let i = 0; i < EMBEDDING_DIM; i++) {
                embedding[i] = outputBuffer.readFloatLE(i * 4);
            }
            return embedding;
        }

        function gpuEmbedBatchTexts(texts) {
            const batchSize = texts.length;
            const textBuffers = texts.map(t => Buffer.from(t, 'utf8'));
            const textLengths = textBuffers.map(b => b.length);
            const outputBuffer = Buffer.alloc(batchSize * EMBEDDING_DIM * 4);

            const result = tomoul_sentence_transformer_gpu_embed_batch(textBuffers, textLengths, batchSize, outputBuffer);
            if (result !== 0) {
                throw new Error(`GPU batch embedding failed with error code: ${result}`);
            }

            const embeddings = [];
            for (let i = 0; i < batchSize; i++) {
                const emb = new Float32Array(EMBEDDING_DIM);
                for (let j = 0; j < EMBEDDING_DIM; j++) {
                    emb[j] = outputBuffer.readFloatLE((i * EMBEDDING_DIM + j) * 4);
                }
                embeddings.push(emb);
            }
            return embeddings;
        }

        // --- GPU warm-up ---
        console.log('\nGPU warm-up run...');
        const gpuWarmup = gpuEmbedText(SENTENCES[0]);
        console.log(`  Embedding dim: ${gpuWarmup.length}`);
        console.log(`  L2 norm: ${l2Norm(gpuWarmup).toFixed(6)}`);

        // --- GPU vs CPU correctness ---
        console.log('\n--- GPU vs CPU Correctness ---');
        let gpuMaxDiff = 0;
        for (let i = 0; i < SENTENCES.length; i++) {
            const gpuEmb = gpuEmbedText(SENTENCES[i]);
            const sim = cosineSimilarity(embeddings[i], gpuEmb);
            let diff = 0;
            for (let j = 0; j < EMBEDDING_DIM; j++) {
                diff = Math.max(diff, Math.abs(embeddings[i][j] - gpuEmb[j]));
            }
            gpuMaxDiff = Math.max(gpuMaxDiff, diff);
            console.log(`  Sentence ${i}: cosine=${sim.toFixed(6)}, max_abs_diff=${diff.toExponential(3)}`);
        }
        console.log(`  Overall max abs diff: ${gpuMaxDiff.toExponential(3)}`);
        console.log(`  Match: ${gpuMaxDiff < 0.05 ? 'PASS' : 'FAIL (divergence > 0.05)'}`);

        // --- GPU single sentence ---
        console.log('\n--- GPU Single Sentence ---');
        times = [];
        for (let i = 0; i < NUM_ITERATIONS; i++) {
            const start = performance.now();
            gpuEmbedText(SENTENCES[0]);
            const end = performance.now();
            times.push(end - start);
        }

        avg = times.reduce((a, b) => a + b, 0) / times.length;
        sorted = [...times].sort((a, b) => a - b);
        p50 = sorted[Math.floor(sorted.length / 2)];
        min = sorted[0];
        max = sorted[sorted.length - 1];

        gpuSingleAvg = avg;
        console.log(`  GPU single sentence (${NUM_ITERATIONS} runs):`);
        console.log(`    Average: ${avg.toFixed(2)}ms`);
        console.log(`    Median:  ${p50.toFixed(2)}ms`);
        console.log(`    Min:     ${min.toFixed(2)}ms`);
        console.log(`    Max:     ${max.toFixed(2)}ms`);
        console.log(`    Speedup vs CPU: ${(singleAvg / avg).toFixed(2)}x`);

        // --- GPU 5 sentences batched ---
        console.log('\n--- GPU 5 Sentences (batched) ---');
        times = [];
        for (let i = 0; i < NUM_ITERATIONS; i++) {
            const start = performance.now();
            gpuEmbedBatchTexts(batch5Texts);
            const end = performance.now();
            times.push(end - start);
        }

        avg = times.reduce((a, b) => a + b, 0) / times.length;
        sorted = [...times].sort((a, b) => a - b);
        p50 = sorted[Math.floor(sorted.length / 2)];
        min = sorted[0];

        gpuBatch5Avg = avg;
        console.log(`  GPU 5 sentences batched (${NUM_ITERATIONS} runs):`);
        console.log(`    Average: ${avg.toFixed(2)}ms  (${(avg / 5).toFixed(2)}ms per sentence)`);
        console.log(`    Median:  ${p50.toFixed(2)}ms  (${(p50 / 5).toFixed(2)}ms per sentence)`);
        console.log(`    Min:     ${min.toFixed(2)}ms  (${(min / 5).toFixed(2)}ms per sentence)`);
        console.log(`    Speedup vs CPU batched: ${(batched5Avg / avg).toFixed(2)}x`);
        console.log(`    Speedup vs CPU sequential: ${(seq5Avg / avg).toFixed(2)}x`);

        // --- GPU 10 sentences batched ---
        console.log('\n--- GPU 10 Sentences (batched) ---');
        times = [];
        for (let i = 0; i < NUM_ITERATIONS; i++) {
            const start = performance.now();
            gpuEmbedBatchTexts(BATCH_SENTENCES);
            const end = performance.now();
            times.push(end - start);
        }

        avg = times.reduce((a, b) => a + b, 0) / times.length;
        sorted = [...times].sort((a, b) => a - b);
        p50 = sorted[Math.floor(sorted.length / 2)];
        min = sorted[0];

        gpuBatch10Avg = avg;
        console.log(`  GPU 10 sentences batched (${NUM_ITERATIONS} runs):`);
        console.log(`    Average: ${avg.toFixed(2)}ms  (${(avg / 10).toFixed(2)}ms per sentence)`);
        console.log(`    Median:  ${p50.toFixed(2)}ms  (${(p50 / 10).toFixed(2)}ms per sentence)`);
        console.log(`    Min:     ${min.toFixed(2)}ms  (${(min / 10).toFixed(2)}ms per sentence)`);
        console.log(`    Speedup vs CPU batched: ${(batched10Avg / avg).toFixed(2)}x`);
        console.log(`    Speedup vs CPU sequential: ${(batch10Avg / avg).toFixed(2)}x`);

        tomoul_sentence_transformer_gpu_destroy();
    }
}

// Cleanup
tomoul_sentence_transformer_destroy();

// Summary
console.log('\n============================================================');
console.log('Summary (Tomoul / Node.js FFI)');
console.log('============================================================');
console.log(`  Variant:            ${VARIANT.toUpperCase()}`);
console.log(`  Single sentence:    ${singleAvg.toFixed(2)}ms avg, ${singleP50.toFixed(2)}ms p50`);
console.log(`  5 sent sequential:  ${seq5Avg.toFixed(2)}ms avg (${(seq5Avg / 5).toFixed(2)}ms/sent)`);
console.log(`  10 sent sequential: ${batch10Avg.toFixed(2)}ms avg (${(batch10Avg / 10).toFixed(2)}ms/sent)`);
console.log(`  5 sent batched:     ${batched5Avg.toFixed(2)}ms avg (${(batched5Avg / 5).toFixed(2)}ms/sent) [${(seq5Avg / batched5Avg).toFixed(2)}x speedup]`);
console.log(`  10 sent batched:    ${batched10Avg.toFixed(2)}ms avg (${(batched10Avg / 10).toFixed(2)}ms/sent) [${(batch10Avg / batched10Avg).toFixed(2)}x speedup]`);
if (gpuAvailable) {
    console.log('  --- GPU ---');
    console.log(`  GPU single:         ${gpuSingleAvg.toFixed(2)}ms avg [${(singleAvg / gpuSingleAvg).toFixed(2)}x vs CPU]`);
    console.log(`  GPU 5 batched:      ${gpuBatch5Avg.toFixed(2)}ms avg (${(gpuBatch5Avg / 5).toFixed(2)}ms/sent) [${(seq5Avg / gpuBatch5Avg).toFixed(2)}x vs CPU seq]`);
    console.log(`  GPU 10 batched:     ${gpuBatch10Avg.toFixed(2)}ms avg (${(gpuBatch10Avg / 10).toFixed(2)}ms/sent) [${(batch10Avg / gpuBatch10Avg).toFixed(2)}x vs CPU seq]`);
}
console.log('============================================================');
