#!/usr/bin/env node
/**
 * Silero VAD Benchmark - Node.js
 *
 * Loads the model once, then runs multiple inference calls to measure latency.
 * Tests with speech, silence, and noise WAV files.
 */

const fs = require('fs');
const path = require('path');
const koffi = require('koffi');

const platform = process.platform;
const arch = process.arch;

let libName;
if (platform === 'linux') {
    libName = 'libtomoul_silero_vad.so';
} else if (platform === 'darwin') {
    libName = 'libtomoul_silero_vad.dylib';
} else {
    console.error(`Unsupported platform: ${platform}`);
    process.exit(1);
}

const PROJECT_ROOT = path.join(__dirname, '../..');

// Try release/lib first, then zig-out/lib
let LIB_PATH = path.join(PROJECT_ROOT, 'release/lib', libName);
if (!fs.existsSync(LIB_PATH)) {
    LIB_PATH = path.join(PROJECT_ROOT, 'zig-out/lib', libName);
}
const WEIGHTS_PATH = path.join(PROJECT_ROOT, 'artifacts/silero_vad.tl');

if (!fs.existsSync(LIB_PATH)) {
    console.error(`Library not found: ${LIB_PATH}`);
    console.error('Build with: zig build lib -Dmodel=silero_vad --release=fast');
    process.exit(1);
}

if (!fs.existsSync(WEIGHTS_PATH)) {
    console.error(`Weights not found: ${WEIGHTS_PATH}`);
    process.exit(1);
}

console.log('============================================================');
console.log('Tomoul Silero VAD Benchmark (Node.js FFI)');
console.log('============================================================\n');

// Load library
const lib = koffi.load(LIB_PATH);
const tomoul_vad_init = lib.func('bool tomoul_vad_init(_In_ uint8_t*, size_t)');
const tomoul_vad_process = lib.func('float tomoul_vad_process(_In_ float*, size_t)');
const tomoul_vad_reset = lib.func('void tomoul_vad_reset()');
const tomoul_vad_free = lib.func('void tomoul_vad_free()');

// Initialize model
console.log('Loading model...');
const loadStart = performance.now();
const weightsData = fs.readFileSync(WEIGHTS_PATH);
const initOk = tomoul_vad_init(weightsData, weightsData.length);
const loadEnd = performance.now();

if (!initOk) {
    console.error('Failed to initialize VAD model');
    process.exit(1);
}
console.log(`Model loaded in ${((loadEnd - loadStart)).toFixed(1)}ms\n`);

// Read WAV file and return f32 samples
function readWav(filePath) {
    const buf = fs.readFileSync(filePath);
    // Skip 44-byte WAV header, read 16-bit PCM
    const samples = [];
    for (let i = 44; i < buf.length - 1; i += 2) {
        const val = buf.readInt16LE(i);
        samples.push(val / 32768.0);
    }
    return new Float32Array(samples);
}

// Process audio in 512-sample chunks and measure timing
function benchmarkAudio(name, samples, numRuns) {
    const CHUNK_SIZE = 512;
    const numChunks = Math.floor(samples.length / CHUNK_SIZE);

    const times = [];
    for (let run = 0; run < numRuns; run++) {
        tomoul_vad_reset();
        const start = performance.now();
        for (let c = 0; c < numChunks; c++) {
            const chunk = samples.slice(c * CHUNK_SIZE, (c + 1) * CHUNK_SIZE);
            tomoul_vad_process(chunk, CHUNK_SIZE);
        }
        const elapsed = performance.now() - start;
        times.push(elapsed);
    }

    times.sort((a, b) => a - b);
    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    const p50 = times[Math.floor(times.length / 2)];
    const perChunk = avg / numChunks;
    const durationMs = (samples.length / 16000) * 1000;
    const rtf = avg / durationMs;

    console.log(`--- ${name} (${numChunks} chunks, ${(durationMs / 1000).toFixed(2)}s audio) ---`);
    console.log(`  Total:     ${avg.toFixed(2)}ms avg, ${p50.toFixed(2)}ms p50`);
    console.log(`  Per chunk: ${perChunk.toFixed(3)}ms (${(CHUNK_SIZE / 16000 * 1000).toFixed(1)}ms audio = 32ms)`);
    console.log(`  RTF:       ${rtf.toFixed(4)} (${(1/rtf).toFixed(0)}× realtime)\n`);

    return { avg, p50, perChunk, rtf, numChunks };
}

// Also benchmark raw chunk throughput (single chunk repeated)
function benchmarkSingleChunk(numIters) {
    const chunk = new Float32Array(512);
    // Fill with simple sine wave
    for (let i = 0; i < 512; i++) {
        chunk[i] = Math.sin(2 * Math.PI * 440 * i / 16000) * 0.5;
    }

    // Warmup
    for (let i = 0; i < 10; i++) {
        tomoul_vad_process(chunk, 512);
    }
    tomoul_vad_reset();

    const times = [];
    for (let i = 0; i < numIters; i++) {
        const start = performance.now();
        tomoul_vad_process(chunk, 512);
        const elapsed = performance.now() - start;
        times.push(elapsed);
    }

    times.sort((a, b) => a - b);
    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    const p50 = times[Math.floor(times.length / 2)];
    const min = times[0];

    console.log(`--- Single Chunk Latency (${numIters} calls) ---`);
    console.log(`  Average: ${avg.toFixed(3)}ms`);
    console.log(`  Median:  ${p50.toFixed(3)}ms`);
    console.log(`  Min:     ${min.toFixed(3)}ms\n`);

    return { avg, p50, min };
}

const NUM_RUNS = parseInt(process.env.BENCH_RUNS || '50');

// Single chunk throughput
console.log('Running benchmarks...\n');
const single = benchmarkSingleChunk(500);

// WAV file benchmarks
const wavDir = __dirname;
const wavFiles = ['speech.wav', 'silence.wav', 'noise.wav'];
const results = {};

for (const wav of wavFiles) {
    const wavPath = path.join(wavDir, wav);
    if (fs.existsSync(wavPath)) {
        const samples = readWav(wavPath);
        results[wav] = benchmarkAudio(wav, samples, NUM_RUNS);
    } else {
        console.log(`Skipping ${wav} (not found)`);
    }
}

console.log('============================================================');
console.log('Summary');
console.log('============================================================');
console.log(`  Single chunk latency: ${single.avg.toFixed(3)}ms avg, ${single.p50.toFixed(3)}ms p50`);
for (const [name, r] of Object.entries(results)) {
    console.log(`  ${name}: ${r.avg.toFixed(2)}ms total, ${r.perChunk.toFixed(3)}ms/chunk, ${(1/r.rtf).toFixed(0)}× RT`);
}
console.log('============================================================');

tomoul_vad_free();
