///! GPU vs CPU model comparison test
///!
///! Loads real F32 sentence transformer weights, runs embedding on both
///! CPU and GPU (Vulkan), and verifies outputs match within tolerance.
///!
///! Run with: zig build test-gpu-model
///!
///! Prerequisites:
///!   python3 tools/export_sentence_transformer.py  (generates artifacts/)
///!   Vulkan ICD available (llvmpipe or discrete GPU)
///!
const std = @import("std");
const model_mod = @import("model");
const SentenceTransformer = model_mod.SentenceTransformer;
const SentenceTransformerModel = model_mod.SentenceTransformerModel;
const tokenizer_mod = @import("tokenizer");
const Tokenizer = tokenizer_mod.Tokenizer;
const GpuModel = @import("gpu_model").GpuModel;

const MODEL_PATH = "artifacts/all_minilm_l6_v2.tl";
const MODEL_Q8K_PATH = "artifacts/all_minilm_l6_v2_q8k.tl";
const MODEL_F16_PATH = "artifacts/all_minilm_l6_v2_f16.tl";
const VOCAB_PATH = "artifacts/all_minilm_l6_v2_vocab.txt";

// Tolerance for GPU vs CPU comparison.
// Floating-point reduction order differs between CPU and GPU so we allow
// a small delta. LayerNorm + 6 transformer layers accumulate some error.
const COSINE_TOLERANCE: f32 = 0.98;
const MAX_ABS_TOLERANCE: f32 = 0.05;

const test_sentences = [_][]const u8{
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning models can run on GPUs for faster inference",
    "Zig is a systems programming language",
    "Vulkan compute shaders enable general-purpose GPU computing",
};

fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;
    for (a, b) |ai, bi| {
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0) return 0;
    return dot / denom;
}

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var max: f32 = 0;
    for (a, b) |ai, bi| {
        const diff = @abs(ai - bi);
        if (diff > max) max = diff;
    }
    return max;
}

// =============================================================================
// T1: GPU vs CPU embedding cosine similarity
// =============================================================================

test "GPU vs CPU: 5 reference sentences cosine_sim >= 0.98" {
    const allocator = std.testing.allocator;

    // Load CPU model
    var st = SentenceTransformer.init(allocator, MODEL_PATH, VOCAB_PATH) catch |err| {
        std.debug.print("\nSkipping: model not found ({}).\n", .{err});
        return;
    };
    defer st.deinit();

    // Load GPU model from the same weights
    var gpu = GpuModel.init(allocator, &st.model) catch |err| {
        std.debug.print("\nSkipping: Vulkan init failed ({}).\n", .{err});
        return;
    };
    defer gpu.deinit();

    std.debug.print("\n", .{});
    var all_pass = true;
    for (test_sentences) |sentence| {
        const cpu_emb = try st.embed(sentence);
        const gpu_emb = try gpu.embed(&st.tokenizer, sentence);

        const sim = cosineSimilarity(&cpu_emb, &gpu_emb);
        const max_diff = maxAbsDiff(&cpu_emb, &gpu_emb);
        const pass = sim >= COSINE_TOLERANCE;
        if (!pass) all_pass = false;

        std.debug.print("  cos={d:.6} max_diff={d:.6} {s} | \"{s}\"\n", .{
            sim,
            max_diff,
            if (pass) "PASS" else "FAIL",
            sentence[0..@min(sentence.len, 50)],
        });
    }
    try std.testing.expect(all_pass);
}

// =============================================================================
// T2: GPU embedding latency benchmark
// =============================================================================

test "GPU vs CPU: latency comparison" {
    const allocator = std.testing.allocator;

    var st = SentenceTransformer.init(allocator, MODEL_PATH, VOCAB_PATH) catch |err| {
        std.debug.print("\nSkipping: model not found ({}).\n", .{err});
        return;
    };
    defer st.deinit();

    var gpu = GpuModel.init(allocator, &st.model) catch |err| {
        std.debug.print("\nSkipping: Vulkan init failed ({}).\n", .{err});
        return;
    };
    defer gpu.deinit();

    const sentence = "Machine learning models can run on GPUs for faster inference";
    const warmup = 2;
    const iterations = 5;

    // Warmup
    for (0..warmup) |_| {
        _ = try st.embed(sentence);
        _ = try gpu.embed(&st.tokenizer, sentence);
    }

    // CPU benchmark
    var cpu_timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        _ = try st.embed(sentence);
    }
    const cpu_ns = cpu_timer.read();
    const cpu_ms = @as(f64, @floatFromInt(cpu_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(iterations));

    // GPU benchmark
    var gpu_timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        _ = try gpu.embed(&st.tokenizer, sentence);
    }
    const gpu_ns = gpu_timer.read();
    const gpu_ms = @as(f64, @floatFromInt(gpu_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(iterations));

    const speedup = cpu_ms / gpu_ms;
    std.debug.print("\n  CPU: {d:.2} ms/embed\n  GPU: {d:.2} ms/embed\n  Speedup: {d:.2}x\n", .{
        cpu_ms,
        gpu_ms,
        speedup,
    });
}

// =============================================================================
// T3: Batched GPU vs single-GPU embedding comparison
// =============================================================================

test "GPU batch: embedBatch matches single embed" {
    const allocator = std.testing.allocator;

    var st = SentenceTransformer.init(allocator, MODEL_PATH, VOCAB_PATH) catch |err| {
        std.debug.print("\nSkipping: model not found ({}).\n", .{err});
        return;
    };
    defer st.deinit();

    var gpu_model = GpuModel.init(allocator, &st.model) catch |err| {
        std.debug.print("\nSkipping: Vulkan init failed ({}).\n", .{err});
        return;
    };
    defer gpu_model.deinit();

    // Get single embeddings
    var single_results: [test_sentences.len][384]f32 = undefined;
    for (test_sentences, 0..) |sentence, i| {
        single_results[i] = try gpu_model.embed(&st.tokenizer, sentence);
    }

    // Get batch embeddings
    var batch_results: [test_sentences.len][384]f32 = undefined;
    try gpu_model.embedBatch(&st.tokenizer, &test_sentences, &batch_results);

    // Compare: batch should match single within very tight tolerance
    // (same computation, same shader, just batched dispatch)
    std.debug.print("\n", .{});
    var all_pass = true;
    for (0..test_sentences.len) |i| {
        const sim = cosineSimilarity(&single_results[i], &batch_results[i]);
        const max_diff = maxAbsDiff(&single_results[i], &batch_results[i]);
        const pass = sim >= 0.999;
        if (!pass) all_pass = false;
        std.debug.print("  batch[{d}] cos={d:.6} max_diff={d:.6} {s}\n", .{
            i, sim, max_diff, if (pass) "PASS" else "FAIL",
        });
    }
    try std.testing.expect(all_pass);
}

// =============================================================================
// T4: Batched GPU vs CPU embedding comparison
// =============================================================================

test "GPU batch vs CPU: cosine_sim >= 0.98" {
    const allocator = std.testing.allocator;

    var st = SentenceTransformer.init(allocator, MODEL_PATH, VOCAB_PATH) catch |err| {
        std.debug.print("\nSkipping: model not found ({}).\n", .{err});
        return;
    };
    defer st.deinit();

    var gpu_model = GpuModel.init(allocator, &st.model) catch |err| {
        std.debug.print("\nSkipping: Vulkan init failed ({}).\n", .{err});
        return;
    };
    defer gpu_model.deinit();

    // GPU batch
    var gpu_results: [test_sentences.len][384]f32 = undefined;
    try gpu_model.embedBatch(&st.tokenizer, &test_sentences, &gpu_results);

    // CPU single
    std.debug.print("\n", .{});
    var all_pass = true;
    for (test_sentences, 0..) |sentence, i| {
        const cpu_emb = try st.embed(sentence);
        const sim = cosineSimilarity(&cpu_emb, &gpu_results[i]);
        const pass = sim >= COSINE_TOLERANCE;
        if (!pass) all_pass = false;
        std.debug.print("  batch_gpu_vs_cpu[{d}] cos={d:.6} {s}\n", .{
            i, sim, if (pass) "PASS" else "FAIL",
        });
    }
    try std.testing.expect(all_pass);
}

// =============================================================================
// T5: Batch latency benchmark
// =============================================================================

test "GPU batch vs CPU batch: latency comparison" {
    const allocator = std.testing.allocator;

    var st = SentenceTransformer.init(allocator, MODEL_PATH, VOCAB_PATH) catch |err| {
        std.debug.print("\nSkipping: model not found ({}).\n", .{err});
        return;
    };
    defer st.deinit();

    var gpu_model = GpuModel.init(allocator, &st.model) catch |err| {
        std.debug.print("\nSkipping: Vulkan init failed ({}).\n", .{err});
        return;
    };
    defer gpu_model.deinit();

    const iterations = 3;
    const batch_size = test_sentences.len;

    // Warmup
    {
        var gpu_results: [test_sentences.len][384]f32 = undefined;
        try gpu_model.embedBatch(&st.tokenizer, &test_sentences, &gpu_results);
        for (test_sentences) |s| _ = try st.embed(s);
    }

    // CPU sequential benchmark
    var cpu_timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        for (test_sentences) |s| _ = try st.embed(s);
    }
    const cpu_ns = cpu_timer.read();
    const cpu_ms = @as(f64, @floatFromInt(cpu_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(iterations));

    // GPU batch benchmark
    var gpu_timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        var gpu_results: [test_sentences.len][384]f32 = undefined;
        try gpu_model.embedBatch(&st.tokenizer, &test_sentences, &gpu_results);
    }
    const gpu_ns = gpu_timer.read();
    const gpu_ms = @as(f64, @floatFromInt(gpu_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(iterations));

    const speedup = cpu_ms / gpu_ms;
    std.debug.print("\n  Batch={d} sentences\n  CPU sequential: {d:.2} ms total\n  GPU batch: {d:.2} ms total\n  Speedup: {d:.2}x\n  GPU per-sentence: {d:.2} ms\n", .{
        batch_size,
        cpu_ms,
        gpu_ms,
        speedup,
        gpu_ms / @as(f64, @floatFromInt(batch_size)),
    });
}

// =============================================================================
// T6: Q8K model GPU inference — dequantized to F32 on init
// =============================================================================

test "GPU Q8K: dequant + embed matches CPU F32 within tolerance" {
    const allocator = std.testing.allocator;

    // Load F32 CPU model for reference embeddings
    var st_f32 = SentenceTransformer.init(allocator, MODEL_PATH, VOCAB_PATH) catch |err| {
        std.debug.print("\nSkipping: F32 model not found ({}).\n", .{err});
        return;
    };
    defer st_f32.deinit();

    // Load Q8K model
    var q8k_model = SentenceTransformerModel.init(allocator, MODEL_Q8K_PATH) catch |err| {
        std.debug.print("\nSkipping: Q8K model not found ({}).\n", .{err});
        return;
    };
    defer q8k_model.deinit();

    // Init GPU model from Q8K weights (dequantizes to F32 for GPU upload)
    var gpu_q8k = GpuModel.init(allocator, &q8k_model) catch |err| {
        std.debug.print("\nSkipping: Vulkan init failed ({}).\n", .{err});
        return;
    };
    defer gpu_q8k.deinit();

    // Compare GPU Q8K vs CPU F32
    // Q8K dequantization introduces small errors, so use looser tolerance
    const q8k_cosine_threshold: f32 = 0.95;

    std.debug.print("\n", .{});
    var all_pass = true;
    for (test_sentences) |sentence| {
        const cpu_emb = try st_f32.embed(sentence);
        const gpu_emb = try gpu_q8k.embed(&st_f32.tokenizer, sentence);

        const sim = cosineSimilarity(&cpu_emb, &gpu_emb);
        const max_diff = maxAbsDiff(&cpu_emb, &gpu_emb);
        const pass = sim >= q8k_cosine_threshold;
        if (!pass) all_pass = false;

        std.debug.print("  Q8K cos={d:.6} max_diff={d:.6} {s} | \"{s}\"\n", .{
            sim, max_diff, if (pass) "PASS" else "FAIL",
            sentence[0..@min(sentence.len, 50)],
        });
    }
    try std.testing.expect(all_pass);
}

// =============================================================================
// T7: F16 model GPU inference — cast to F32 on init
// =============================================================================

test "GPU F16: cast + embed matches CPU F32 within tolerance" {
    const allocator = std.testing.allocator;

    var st_f32 = SentenceTransformer.init(allocator, MODEL_PATH, VOCAB_PATH) catch |err| {
        std.debug.print("\nSkipping: F32 model not found ({}).\n", .{err});
        return;
    };
    defer st_f32.deinit();

    var f16_model = SentenceTransformerModel.init(allocator, MODEL_F16_PATH) catch |err| {
        std.debug.print("\nSkipping: F16 model not found ({}).\n", .{err});
        return;
    };
    defer f16_model.deinit();

    var gpu_f16 = GpuModel.init(allocator, &f16_model) catch |err| {
        std.debug.print("\nSkipping: Vulkan init failed ({}).\n", .{err});
        return;
    };
    defer gpu_f16.deinit();

    // F16→F32 precision loss is small, so threshold is tighter than Q8K
    const f16_cosine_threshold: f32 = 0.97;

    std.debug.print("\n", .{});
    var all_pass = true;
    for (test_sentences) |sentence| {
        const cpu_emb = try st_f32.embed(sentence);
        const gpu_emb = try gpu_f16.embed(&st_f32.tokenizer, sentence);

        const sim = cosineSimilarity(&cpu_emb, &gpu_emb);
        const max_diff = maxAbsDiff(&cpu_emb, &gpu_emb);
        const pass = sim >= f16_cosine_threshold;
        if (!pass) all_pass = false;

        std.debug.print("  F16 cos={d:.6} max_diff={d:.6} {s} | \"{s}\"\n", .{
            sim, max_diff, if (pass) "PASS" else "FAIL",
            sentence[0..@min(sentence.len, 50)],
        });
    }
    try std.testing.expect(all_pass);
}
