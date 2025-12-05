#!/usr/bin/env node
/**
 * Punctuation Benchmark - Float32 vs Q8_0 Comparison
 *
 * Compares inference speed and model size between float32 and Q8_0 quantized models.
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

// Model paths
const F32_WEIGHTS = path.join(PROJECT_ROOT, 'release/fullstop_punctuation_multilang_large.tl');
const Q8_WEIGHTS = path.join(PROJECT_ROOT, 'artifacts/fullstop_punctuation_multilang_large_q8.tl');
const VOCAB_PATH = path.join(PROJECT_ROOT, 'artifacts/fullstop_punctuation_multilang_large_vocab.txt');

// Check files exist
if (!fs.existsSync(LIB_PATH)) {
    console.error(`Library not found: ${LIB_PATH}`);
    console.error('Run: zig build lib');
    process.exit(1);
}

console.log('============================================================');
console.log('Tomoul Punctuation Benchmark - Float32 vs Q8_0 Comparison');
console.log('============================================================\n');

// File sizes
const f32Size = fs.existsSync(F32_WEIGHTS) ? fs.statSync(F32_WEIGHTS).size : 0;
const q8Size = fs.existsSync(Q8_WEIGHTS) ? fs.statSync(Q8_WEIGHTS).size : 0;

console.log('Model Sizes:');
console.log(`  Float32: ${(f32Size / 1024 / 1024).toFixed(1)} MB`);
console.log(`  Q8_0:    ${(q8Size / 1024 / 1024).toFixed(1)} MB`);
console.log(`  Compression: ${(f32Size / q8Size).toFixed(2)}x\n`);

// Load library
const lib = koffi.load(LIB_PATH);
const tomoul_xlm_roberta_punctuation_init = lib.func('int tomoul_xlm_roberta_punctuation_init(string, string)');
const tomoul_xlm_roberta_punctuation_process = lib.func('int tomoul_xlm_roberta_punctuation_process(_In_ uint8_t*, size_t, _Out_ uint8_t*, size_t)');
const tomoul_xlm_roberta_punctuation_destroy = lib.func('void tomoul_xlm_roberta_punctuation_destroy()');
const tomoul_xlm_roberta_punctuation_version = lib.func('string tomoul_xlm_roberta_punctuation_version()');
const tomoul_xlm_roberta_punctuation_quant_format = lib.func('int tomoul_xlm_roberta_punctuation_quant_format()');

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

// Test text
const testText = "hello world this is a test of the punctuation restoration system we are going to see how fast it can process this paragraph which contains exactly one hundred words or close to it the goal is to measure the performance difference between the python implementation using pytorch and the zig implementation using native code with manual memory management this benchmark will help us understand if the zero interpreter overhead and better memory control in zig provides a significant speedup compared to python we expect zig to be two to five times faster than python for this workload lets see if that prediction holds true";

const NUM_ITERATIONS = 10;

async function runBenchmark(name, weightsPath) {
    console.log(`\n--- ${name} ---`);

    if (!fs.existsSync(weightsPath)) {
        console.log(`  SKIPPED: Model file not found: ${weightsPath}`);
        return null;
    }

    // Initialize
    console.log('Loading model...');
    const loadStart = performance.now();
    const initResult = tomoul_xlm_roberta_punctuation_init(weightsPath, VOCAB_PATH);
    const loadEnd = performance.now();

    if (initResult !== 0) {
        console.log(`  FAILED: Init error code ${initResult}`);
        return null;
    }

    const loadTime = loadEnd - loadStart;
    console.log(`  Load time: ${(loadTime / 1000).toFixed(2)}s`);

    // Check quant format
    const quantFormat = tomoul_xlm_roberta_punctuation_quant_format();
    console.log(`  Quant format: ${quantFormat === 1 ? 'float32' : quantFormat === 2 ? 'Q8_0' : 'unknown'}`);

    // Warm-up
    processText(testText);

    // Benchmark
    const times = [];
    let result = '';
    for (let i = 0; i < NUM_ITERATIONS; i++) {
        const start = performance.now();
        result = processText(testText);
        const end = performance.now();
        times.push(end - start);
    }

    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    const min = Math.min(...times);
    const max = Math.max(...times);

    console.log(`  Inference (${NUM_ITERATIONS} runs):`);
    console.log(`    Average: ${avg.toFixed(2)}ms`);
    console.log(`    Min: ${min.toFixed(2)}ms`);
    console.log(`    Max: ${max.toFixed(2)}ms`);
    console.log(`  Output preview: "${result.substring(0, 80)}..."`);

    // Cleanup
    tomoul_xlm_roberta_punctuation_destroy();

    return { loadTime, avg, min, max };
}

async function main() {
    const results = {};

    // Run float32 benchmark
    results.f32 = await runBenchmark('Float32 Model', F32_WEIGHTS);

    // Run Q8_0 benchmark
    results.q8 = await runBenchmark('Q8_0 Quantized Model', Q8_WEIGHTS);

    // Summary
    console.log('\n============================================================');
    console.log('Summary');
    console.log('============================================================');

    if (results.f32 && results.q8) {
        console.log(`\nModel Size: Q8_0 is ${(f32Size / q8Size).toFixed(2)}x smaller`);
        console.log(`Load Time: Q8_0 is ${(results.f32.loadTime / results.q8.loadTime).toFixed(2)}x faster`);
        console.log(`Inference: Q8_0 is ${(results.f32.avg / results.q8.avg).toFixed(2)}x ${results.q8.avg < results.f32.avg ? 'faster' : 'slower'}`);
    } else if (results.q8) {
        console.log('\nOnly Q8_0 model was tested.');
    } else if (results.f32) {
        console.log('\nOnly Float32 model was tested.');
    } else {
        console.log('\nNo models could be loaded.');
    }
}

main().catch(console.error);
