///! Validation Tests for Sentence Transformer (all-MiniLM-L6-v2)
///!
///! Go/no-go gate: cosine similarity ≥ 0.99 for all 5 reference sentences.
///!
///! Run with: zig build test-sentence-transformer
///!
///! Prerequisites:
///!   python3 tools/export_sentence_transformer.py   (generates artifacts/)
///!
const std = @import("std");
const model_mod = @import("model");
const SentenceTransformer = model_mod.SentenceTransformer;
const SentenceTransformerModel = model_mod.SentenceTransformerModel;

const tokenizer_mod = @import("tokenizer");
const Tokenizer = tokenizer_mod.Tokenizer;
const SpecialTokens = tokenizer_mod.SpecialTokens;

const MODEL_PATH = "artifacts/all_minilm_l6_v2.tl";
const VOCAB_PATH = "artifacts/all_minilm_l6_v2_vocab.txt";
const REFERENCE_PATH = "artifacts/minilm_reference_embeddings.json";

// =============================================================================
// Reference Data (parsed from JSON at test time)
// =============================================================================

const ReferenceData = struct {
    sentences: []const []const u8,
    embeddings: []const []const f32,
    token_ids: []const []const u32,
};

fn parseReferenceJson(allocator: std.mem.Allocator) !struct { data: ReferenceData, arena: *std.heap.ArenaAllocator } {
    const file = std.fs.cwd().openFile(REFERENCE_PATH, .{}) catch |err| {
        std.debug.print("\nSkipping: reference file not found. Run 'python3 tools/export_sentence_transformer.py' first.\n", .{});
        return err;
    };
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 256 * 1024 * 1024);
    defer allocator.free(content);

    var arena = try allocator.create(std.heap.ArenaAllocator);
    arena.* = std.heap.ArenaAllocator.init(allocator);
    const aa = arena.allocator();

    const parsed = try std.json.parseFromSlice(std.json.Value, aa, content, .{});
    const root = parsed.value.object;

    // Parse sentences
    const sentences_json = root.get("sentences").?.array;
    const sentences = try aa.alloc([]const u8, sentences_json.items.len);
    for (sentences_json.items, 0..) |s, i| {
        sentences[i] = s.string;
    }

    // Parse embeddings
    const embeddings_json = root.get("embeddings").?.array;
    const embeddings = try aa.alloc([]const f32, embeddings_json.items.len);
    for (embeddings_json.items, 0..) |emb_json, i| {
        const vals = emb_json.array;
        const emb = try aa.alloc(f32, vals.items.len);
        for (vals.items, 0..) |v, j| {
            emb[j] = switch (v) {
                .float => @floatCast(v.float),
                .integer => @floatFromInt(v.integer),
                else => 0.0,
            };
        }
        embeddings[i] = emb;
    }

    // Parse tokenizer_reference
    const tok_ref_json = root.get("tokenizer_reference").?.array;
    const token_ids = try aa.alloc([]const u32, tok_ref_json.items.len);
    for (tok_ref_json.items, 0..) |entry, i| {
        const ids_json = entry.object.get("token_ids").?.array;
        const ids = try aa.alloc(u32, ids_json.items.len);
        for (ids_json.items, 0..) |v, j| {
            ids[j] = @intCast(v.integer);
        }
        token_ids[i] = ids;
    }

    return .{
        .data = .{
            .sentences = sentences,
            .embeddings = embeddings,
            .token_ids = token_ids,
        },
        .arena = arena,
    };
}

// =============================================================================
// Helpers
// =============================================================================

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

// =============================================================================
// T5 Core: Embedding accuracy (go/no-go gate)
// =============================================================================

test "Embedding: 5 reference sentences cosine_sim >= 0.99" {
    const allocator = std.testing.allocator;

    const ref_result = parseReferenceJson(allocator) catch return;
    defer {
        ref_result.arena.deinit();
        allocator.destroy(ref_result.arena);
    }
    const ref = ref_result.data;

    var st = SentenceTransformer.init(allocator, MODEL_PATH, VOCAB_PATH) catch |err| {
        std.debug.print("\nSkipping: model not found ({}).\n", .{err});
        return;
    };
    defer st.deinit();

    std.debug.print("\n", .{});
    var all_pass = true;
    for (ref.sentences, ref.embeddings, 0..) |sentence, ref_emb, i| {
        const zig_emb = try st.embed(sentence);
        const sim = cosineSimilarity(&zig_emb, ref_emb);
        const pass = sim >= 0.99;
        if (!pass) all_pass = false;
        std.debug.print("  Sentence {d}: cosine_sim = {d:.6} {s} | \"{s}\"\n", .{
            i,
            sim,
            if (pass) "PASS" else "FAIL",
            sentence[0..@min(sentence.len, 50)],
        });
    }
    try std.testing.expect(all_pass);
}

// =============================================================================
// Tokenizer correctness
// =============================================================================

test "Tokenizer: known input → known token IDs" {
    const allocator = std.testing.allocator;

    const ref_result = parseReferenceJson(allocator) catch return;
    defer {
        ref_result.arena.deinit();
        allocator.destroy(ref_result.arena);
    }
    const ref = ref_result.data;

    var tokenizer = Tokenizer.init(allocator, VOCAB_PATH) catch |err| {
        std.debug.print("\nSkipping: vocab not found ({}).\n", .{err});
        return;
    };
    defer tokenizer.deinit();

    std.debug.print("\n", .{});
    for (ref.sentences, ref.token_ids, 0..) |sentence, expected_ids, i| {
        var output = try tokenizer.encode(sentence);
        defer output.deinit(allocator);

        var match = output.input_ids.len == expected_ids.len;
        if (match) {
            for (output.input_ids, expected_ids) |got, exp| {
                if (got != exp) {
                    match = false;
                    break;
                }
            }
        }

        if (!match) {
            std.debug.print("  Sentence {d} MISMATCH:\n", .{i});
            std.debug.print("    expected ({d}): ", .{expected_ids.len});
            for (expected_ids) |id| std.debug.print("{d} ", .{id});
            std.debug.print("\n    got      ({d}): ", .{output.input_ids.len});
            for (output.input_ids) |id| std.debug.print("{d} ", .{id});
            std.debug.print("\n", .{});
        } else {
            std.debug.print("  Sentence {d}: {d} tokens match\n", .{ i, expected_ids.len });
        }

        try std.testing.expectEqual(expected_ids.len, output.input_ids.len);
        for (output.input_ids, expected_ids) |got, exp| {
            try std.testing.expectEqual(exp, got);
        }
    }
}

// =============================================================================
// Edge cases
// =============================================================================

test "Tokenizer: empty string produces CLS + SEP" {
    const allocator = std.testing.allocator;

    var tokenizer = Tokenizer.init(allocator, VOCAB_PATH) catch |err| {
        std.debug.print("\nSkipping: vocab not found ({}).\n", .{err});
        return;
    };
    defer tokenizer.deinit();

    var output = try tokenizer.encode("");
    defer output.deinit(allocator);

    // Empty string should still produce [CLS] + [SEP]
    try std.testing.expectEqual(@as(usize, 2), output.input_ids.len);
    try std.testing.expectEqual(SpecialTokens.CLS, output.input_ids[0]);
    try std.testing.expectEqual(SpecialTokens.SEP, output.input_ids[1]);
}

test "Embedding: empty string produces valid vector" {
    const allocator = std.testing.allocator;

    var st = SentenceTransformer.init(allocator, MODEL_PATH, VOCAB_PATH) catch |err| {
        std.debug.print("\nSkipping: model not found ({}).\n", .{err});
        return;
    };
    defer st.deinit();

    const emb = try st.embed("");

    // Should produce a valid normalized vector (L2 norm ≈ 1.0)
    var norm_sq: f32 = 0;
    for (emb) |v| norm_sq += v * v;
    const norm = @sqrt(norm_sq);
    std.debug.print("\nEmpty string embedding L2 norm: {d:.6}\n", .{norm});
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 0.01);
}

test "Batch consistency: embedBatch matches individual embed" {
    const allocator = std.testing.allocator;

    var st = SentenceTransformer.init(allocator, MODEL_PATH, VOCAB_PATH) catch |err| {
        std.debug.print("\nSkipping: model not found ({}).\n", .{err});
        return;
    };
    defer st.deinit();

    const texts = [_][]const u8{ "hello world", "testing batch" };

    // Individual embeddings
    const emb_a = try st.embed(texts[0]);
    const emb_b = try st.embed(texts[1]);

    // Batch embeddings
    const batch = try st.embedBatch(&texts);
    defer allocator.free(batch);

    // Must be identical (deterministic)
    for (0..384) |j| {
        try std.testing.expectEqual(emb_a[j], batch[0][j]);
        try std.testing.expectEqual(emb_b[j], batch[1][j]);
    }
}

test "Batch consistency Q8K: embedBatch matches individual embed" {
    const allocator = std.testing.allocator;
    const Q8K_MODEL_PATH = "artifacts/all_minilm_l6_v2_q8k.tl";

    var st = SentenceTransformer.init(allocator, Q8K_MODEL_PATH, VOCAB_PATH) catch |err| {
        std.debug.print("\nSkipping: Q8K model not found ({}).\n", .{err});
        return;
    };
    defer st.deinit();

    // Use enough sentences with enough tokens to trigger blocked path (M > 32)
    // 5 sentences × ~10 tokens each → total_tokens ≈ 50-60 (> SKINNY_M_THRESHOLD=32)
    const texts = [_][]const u8{
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "I had pizza for lunch yesterday",
        "The capital of France is Paris",
        "Quantum computing uses qubits instead of classical bits",
    };

    // Individual embeddings
    var singles: [texts.len][384]f32 = undefined;
    for (0..texts.len) |i| {
        singles[i] = try st.embed(texts[i]);
    }

    // Batch embeddings
    const batch = try st.embedBatch(&texts);
    defer allocator.free(batch);

    std.debug.print("\nQ8K batch vs sequential ({d} sentences):\n", .{texts.len});
    for (0..texts.len) |i| {
        var dot: f32 = 0;
        var norm_s: f32 = 0;
        var norm_b: f32 = 0;
        var max_diff: f32 = 0;
        for (0..384) |j| {
            dot += singles[i][j] * batch[i][j];
            norm_s += singles[i][j] * singles[i][j];
            norm_b += batch[i][j] * batch[i][j];
            const d = @abs(singles[i][j] - batch[i][j]);
            if (d > max_diff) max_diff = d;
        }
        const cosine = dot / (@sqrt(norm_s) * @sqrt(norm_b));
        std.debug.print("  Sentence {d}: cosine={d:.6}, max_diff={d:.6}\n", .{ i, cosine, max_diff });
        try std.testing.expect(cosine > 0.99);
    }
}

// =============================================================================
// Benchmark (print, don't assert)
// =============================================================================

test "Benchmark: single sentence latency" {
    const allocator = std.testing.allocator;

    var st = SentenceTransformer.init(allocator, MODEL_PATH, VOCAB_PATH) catch |err| {
        std.debug.print("\nSkipping benchmark: model not found ({}).\n", .{err});
        return;
    };
    defer st.deinit();

    // Warmup
    _ = try st.embed("warmup sentence");

    const iterations: usize = 20;
    var timer = try std.time.Timer.start();

    for (0..iterations) |_| {
        _ = try st.embed("The quick brown fox jumps over the lazy dog");
    }

    const elapsed_ns = timer.read();
    const avg_ms = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations)) / 1_000_000.0;
    std.debug.print("\nSingle sentence latency: {d:.2} ms (avg over {d} iterations)\n", .{ avg_ms, iterations });
}
