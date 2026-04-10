///! Validation Tests for Gemma Model
///!
///! Tests config parameters, tokenizer encode/decode, and GQA config derivation.
///!
///! Run with: zig build test-gemma
///!
const std = @import("std");
const model_mod = @import("model");
const config_mod = model_mod.config;
const tokenizer_mod = model_mod.tokenizer;

const GemmaConfig = config_mod.GemmaConfig;
const GemmaVariant = config_mod.GemmaVariant;
const GemmaTokens = config_mod.GemmaTokens;
const Tokenizer = tokenizer_mod.Tokenizer;

// =============================================================================
// Config Tests
// =============================================================================

test "gemma_2b variant config" {
    const cfg = GemmaConfig.forVariant(.gemma_2b);
    try std.testing.expectEqual(@as(usize, 18), cfg.num_layers);
    try std.testing.expectEqual(@as(usize, 2048), cfg.hidden_dim);
    try std.testing.expectEqual(@as(usize, 8), cfg.num_heads);
    try std.testing.expectEqual(@as(usize, 1), cfg.num_kv_heads);
    try std.testing.expectEqual(@as(usize, 256), cfg.head_dim);
    try std.testing.expectEqual(@as(usize, 16384), cfg.intermediate_dim);
    try std.testing.expectEqual(@as(usize, 256000), cfg.vocab_size);
    try std.testing.expectEqual(@as(usize, 8192), cfg.max_position);
    try std.testing.expect(!cfg.isGemma2());
}

test "gemma_7b variant config" {
    const cfg = GemmaConfig.forVariant(.gemma_7b);
    try std.testing.expectEqual(@as(usize, 28), cfg.num_layers);
    try std.testing.expectEqual(@as(usize, 3072), cfg.hidden_dim);
    try std.testing.expectEqual(@as(usize, 16), cfg.num_heads);
    try std.testing.expectEqual(@as(usize, 16), cfg.num_kv_heads);
    try std.testing.expect(!cfg.isGemma2());
}

test "gemma2_9b variant config" {
    const cfg = GemmaConfig.forVariant(.gemma2_9b);
    try std.testing.expectEqual(@as(usize, 42), cfg.num_layers);
    try std.testing.expectEqual(@as(usize, 3584), cfg.hidden_dim);
    try std.testing.expectEqual(@as(usize, 16), cfg.num_heads);
    try std.testing.expectEqual(@as(usize, 8), cfg.num_kv_heads);
    try std.testing.expect(cfg.isGemma2());
}

test "gemma2_27b variant config" {
    const cfg = GemmaConfig.forVariant(.gemma2_27b);
    try std.testing.expectEqual(@as(usize, 46), cfg.num_layers);
    try std.testing.expectEqual(@as(usize, 4608), cfg.hidden_dim);
    try std.testing.expectEqual(@as(usize, 32), cfg.num_heads);
    try std.testing.expectEqual(@as(usize, 16), cfg.num_kv_heads);
    try std.testing.expectEqual(@as(usize, 128), cfg.head_dim);
    try std.testing.expect(cfg.isGemma2());
}

test "gqa config derivation" {
    const cfg = GemmaConfig.forVariant(.gemma_2b);
    const gqa = cfg.gqaConfig();
    try std.testing.expectEqual(@as(usize, 8), gqa.num_heads);
    try std.testing.expectEqual(@as(usize, 1), gqa.num_kv_heads);
    try std.testing.expectEqual(@as(usize, 256), gqa.head_dim);
    try std.testing.expectEqual(@as(usize, 2048), gqa.hidden_dim);
    try std.testing.expectEqual(@as(usize, 256), gqa.kv_dim()); // 1 * 256
    try std.testing.expectEqual(@as(usize, 8), gqa.heads_per_group()); // 8 / 1
}

test "gqa config — multi-kv-head variant" {
    const cfg = GemmaConfig.forVariant(.gemma2_9b);
    const gqa = cfg.gqaConfig();
    try std.testing.expectEqual(@as(usize, 2048), gqa.kv_dim()); // 8 * 256
    try std.testing.expectEqual(@as(usize, 2), gqa.heads_per_group()); // 16 / 8
}

test "kv dim matches gqa kv_dim" {
    const cfg = GemmaConfig.forVariant(.gemma_2b);
    try std.testing.expectEqual(cfg.kvDim(), cfg.gqaConfig().kv_dim());
}

test "embedding scale" {
    const cfg = GemmaConfig.forVariant(.gemma_2b);
    const scale = cfg.embeddingScale();
    // sqrt(2048) ≈ 45.2548
    try std.testing.expectApproxEqAbs(@as(f32, 45.2548), scale, 0.001);
}

test "special tokens" {
    try std.testing.expectEqual(@as(u32, 0), GemmaTokens.PAD);
    try std.testing.expectEqual(@as(u32, 1), GemmaTokens.EOS);
    try std.testing.expectEqual(@as(u32, 2), GemmaTokens.BOS);
    try std.testing.expectEqual(@as(u32, 3), GemmaTokens.UNK);
}

// =============================================================================
// Tokenizer Tests (using synthetic binary vocab)
// =============================================================================

/// Build a minimal binary vocab for testing.
/// Vocab: 0="<pad>", 1="<eos>", 2="<bos>", 3="<unk>", 4="▁H", 5="e", 6="l", 7="o", 8="▁w", 9="r", 10="d", 11="▁", 12="Hello", 13="llo", 14="▁world"
fn buildTestVocab(allocator: std.mem.Allocator) ![]u8 {
    const tokens = [_]struct { text: []const u8, score: f32 }{
        .{ .text = "<pad>", .score = 0.0 }, // 0
        .{ .text = "<eos>", .score = 0.0 }, // 1
        .{ .text = "<bos>", .score = 0.0 }, // 2
        .{ .text = "<unk>", .score = 0.0 }, // 3
        .{ .text = "\xE2\x96\x81H", .score = -1.0 }, // 4: ▁H
        .{ .text = "e", .score = -2.0 }, // 5
        .{ .text = "l", .score = -2.0 }, // 6
        .{ .text = "o", .score = -2.0 }, // 7
        .{ .text = "\xE2\x96\x81w", .score = -1.0 }, // 8: ▁w
        .{ .text = "r", .score = -2.0 }, // 9
        .{ .text = "d", .score = -2.0 }, // 10
        .{ .text = "\xE2\x96\x81", .score = -3.0 }, // 11: ▁ (bare word boundary)
        .{ .text = "ello", .score = -0.5 }, // 12: merged — higher score = merged first
        .{ .text = "llo", .score = -1.5 }, // 13
        .{ .text = "\xE2\x96\x81world", .score = -0.1 }, // 14: ▁world — highest merge priority
        .{ .text = "or", .score = -1.0 }, // 15
        .{ .text = "orld", .score = -0.3 }, // 16
    };

    const vocab_size: u32 = tokens.len;
    const max_token_len: u32 = 16;

    // Calculate total size
    var total_size: usize = 8; // vocab_size + max_token_len
    for (tokens) |t| {
        total_size += 8 + t.text.len; // score(4) + len(4) + bytes
    }

    var buf = try allocator.alloc(u8, total_size);
    var pos: usize = 0;

    // Write header
    std.mem.writeInt(u32, buf[pos..][0..4], vocab_size, .little);
    pos += 4;
    std.mem.writeInt(u32, buf[pos..][0..4], max_token_len, .little);
    pos += 4;

    // Write each token
    for (tokens) |t| {
        std.mem.writeInt(u32, buf[pos..][0..4], @bitCast(t.score), .little);
        pos += 4;
        std.mem.writeInt(u32, buf[pos..][0..4], @intCast(t.text.len), .little);
        pos += 4;
        @memcpy(buf[pos..][0..t.text.len], t.text);
        pos += t.text.len;
    }

    return buf;
}

test "tokenizer — load from bytes" {
    const allocator = std.testing.allocator;
    const vocab_data = try buildTestVocab(allocator);
    defer allocator.free(vocab_data);

    var tok = try Tokenizer.initFromBytes(allocator, vocab_data);
    defer tok.deinit();

    try std.testing.expectEqual(@as(u32, 17), tok.vocab_size);
    try std.testing.expectEqual(@as(u32, 16), tok.max_token_len);
}

test "tokenizer — decode skips special tokens" {
    const allocator = std.testing.allocator;
    const vocab_data = try buildTestVocab(allocator);
    defer allocator.free(vocab_data);

    var tok = try Tokenizer.initFromBytes(allocator, vocab_data);
    defer tok.deinit();

    // BOS + "▁H" + EOS — should decode to just "H" (BOS/EOS skipped, ▁ → space → trimmed)
    const ids = [_]u32{ 2, 4, 1 };
    const text = try tok.decode(&ids);
    defer allocator.free(text);

    try std.testing.expectEqualStrings("H", text);
}

test "tokenizer — decode replaces sentencepiece marker" {
    const allocator = std.testing.allocator;
    const vocab_data = try buildTestVocab(allocator);
    defer allocator.free(vocab_data);

    var tok = try Tokenizer.initFromBytes(allocator, vocab_data);
    defer tok.deinit();

    // "▁H" "e" "l" "l" "o" "▁w" "o" "r" "l" "d" → "Hello world"
    const ids = [_]u32{ 4, 5, 6, 6, 7, 8, 7, 9, 6, 10 };
    const text = try tok.decode(&ids);
    defer allocator.free(text);

    try std.testing.expectEqualStrings("Hello world", text);
}

test "tokenizer — encode empty string returns BOS only" {
    const allocator = std.testing.allocator;
    const vocab_data = try buildTestVocab(allocator);
    defer allocator.free(vocab_data);

    var tok = try Tokenizer.initFromBytes(allocator, vocab_data);
    defer tok.deinit();

    const ids = try tok.encode("");
    defer allocator.free(ids);

    try std.testing.expectEqual(@as(usize, 1), ids.len);
    try std.testing.expectEqual(GemmaTokens.BOS, ids[0]);
}

test "tokenizer — encode roundtrip" {
    const allocator = std.testing.allocator;
    const vocab_data = try buildTestVocab(allocator);
    defer allocator.free(vocab_data);

    var tok = try Tokenizer.initFromBytes(allocator, vocab_data);
    defer tok.deinit();

    // Encode "Hello world"
    const ids = try tok.encode("Hello world");
    defer allocator.free(ids);

    // First token must be BOS
    try std.testing.expectEqual(GemmaTokens.BOS, ids[0]);

    // Decode back (skip BOS for comparison)
    const text = try tok.decode(ids);
    defer allocator.free(text);

    try std.testing.expectEqualStrings("Hello world", text);
}

test "tokenizer — encode applies BPE merges" {
    const allocator = std.testing.allocator;
    const vocab_data = try buildTestVocab(allocator);
    defer allocator.free(vocab_data);

    var tok = try Tokenizer.initFromBytes(allocator, vocab_data);
    defer tok.deinit();

    // Encode "Hello world" — BPE should merge characters into larger tokens
    const ids = try tok.encode("Hello world");
    defer allocator.free(ids);

    // Should have fewer tokens than character-level (BOS + 10 chars = 11)
    // BPE should produce something like: BOS, ▁H, ello, ▁world (4 tokens)
    try std.testing.expect(ids.len < 11);
    try std.testing.expect(ids.len >= 2); // At least BOS + one token
}

test "tokenizer — invalid data" {
    const allocator = std.testing.allocator;

    // Too short
    const short_data = [_]u8{ 0, 0, 0, 0 };
    const result = Tokenizer.initFromBytes(allocator, &short_data);
    try std.testing.expectError(error.InvalidTokenizerData, result);
}
