// src/models/gemma/tokenizer.zig
// SentencePiece BPE Tokenizer for Gemma
//
// Loads vocabulary and scores from a binary file exported by tools/export_gemma.py.
// Supports encode (text → token IDs) and decode (token IDs → text).
//
// Binary format:
//   u32: vocab_size
//   u32: max_token_len
//   For each token (vocab_size times):
//     f32: score (BPE merge priority — lower score = later merge = higher priority token)
//     u32: token_len
//     [token_len]u8: token bytes
//
// The BPE merge priority is encoded in the scores: tokens with higher scores
// are merged first during encoding.

const std = @import("std");
const config = @import("config.zig");

/// Special tokens
pub const SpecialTokens = config.GemmaTokens;

/// SentencePiece BPE tokenizer
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: [][]const u8, // id → token string
    scores: []f32, // id → merge score
    vocab_size: u32,
    max_token_len: u32,

    // Reverse lookup: token string → id
    token_to_id: std.StringHashMap(u32),

    const Self = @This();

    /// Load tokenizer from binary file path
    pub fn init(allocator: std.mem.Allocator, path: []const u8) !Self {
        const file = std.fs.cwd().openFile(path, .{}) catch |err| {
            std.debug.print("Error opening tokenizer file '{s}': {}\n", .{ path, err });
            return err;
        };
        defer file.close();

        const content = try file.readToEndAlloc(allocator, 256 * 1024 * 1024); // up to 256MB
        defer allocator.free(content);

        return Self.initFromBytes(allocator, content);
    }

    /// Load tokenizer from in-memory bytes
    pub fn initFromBytes(allocator: std.mem.Allocator, data: []const u8) !Self {
        if (data.len < 8) return error.InvalidTokenizerData;

        var pos: usize = 0;

        const vocab_size = std.mem.readInt(u32, data[pos..][0..4], .little);
        pos += 4;
        const max_token_len = std.mem.readInt(u32, data[pos..][0..4], .little);
        pos += 4;

        var vocab = try allocator.alloc([]const u8, vocab_size);
        var vocab_loaded: u32 = 0;
        errdefer {
            for (vocab[0..vocab_loaded]) |t| allocator.free(t);
            allocator.free(vocab);
        }

        var scores = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(scores);

        var token_to_id = std.StringHashMap(u32).init(allocator);
        errdefer token_to_id.deinit();

        for (0..vocab_size) |i| {
            if (pos + 8 > data.len) return error.InvalidTokenizerData;

            // Read score (f32, little-endian)
            scores[i] = @bitCast(std.mem.readInt(u32, data[pos..][0..4], .little));
            pos += 4;

            // Read token length
            const token_len = std.mem.readInt(u32, data[pos..][0..4], .little);
            pos += 4;

            if (pos + token_len > data.len) return error.InvalidTokenizerData;

            // Copy token string
            const token_str = try allocator.alloc(u8, token_len);
            @memcpy(token_str, data[pos..][0..token_len]);
            pos += token_len;

            vocab[i] = token_str;
            vocab_loaded += 1;

            try token_to_id.put(token_str, @intCast(i));
        }

        return Self{
            .allocator = allocator,
            .vocab = vocab,
            .scores = scores,
            .vocab_size = vocab_size,
            .max_token_len = max_token_len,
            .token_to_id = token_to_id,
        };
    }

    /// Encode text into token IDs using BPE.
    /// Returns allocated slice of token IDs (caller owns memory).
    pub fn encode(self: *const Self, text: []const u8) ![]u32 {
        const allocator = self.allocator;

        if (text.len == 0) {
            const result = try allocator.alloc(u32, 1);
            result[0] = SpecialTokens.BOS;
            return result;
        }

        // Step 1: Initialize with individual UTF-8 characters as tokens
        var tokens: std.ArrayListUnmanaged(u32) = .{};
        errdefer tokens.deinit(allocator);

        // Add BOS token
        try tokens.append(allocator, SpecialTokens.BOS);

        // Convert each UTF-8 codepoint to its token ID
        // SentencePiece uses ▁ (U+2581) as the word boundary marker
        var i: usize = 0;
        var at_word_start = true;
        while (i < text.len) {
            const cp_len = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
            const end = @min(i + cp_len, text.len);
            const char_bytes = text[i..end];

            if (at_word_start and char_bytes[0] != ' ') {
                // Try "▁" + char as a single token (SentencePiece convention)
                // ▁ is 0xE2 0x96 0x81 in UTF-8
                var buf: [8]u8 = undefined;
                buf[0] = 0xE2;
                buf[1] = 0x96;
                buf[2] = 0x81;
                const char_len = end - i;
                if (char_len + 3 <= buf.len) {
                    @memcpy(buf[3..][0..char_len], char_bytes);
                    if (self.token_to_id.get(buf[0 .. 3 + char_len])) |id| {
                        try tokens.append(allocator, id);
                        i = end;
                        at_word_start = false;
                        continue;
                    }
                }
            }

            if (char_bytes[0] == ' ') {
                at_word_start = true;
                i = end;
                continue;
            }

            // Look up single character token
            if (self.token_to_id.get(char_bytes)) |id| {
                try tokens.append(allocator, id);
            } else {
                try tokens.append(allocator, SpecialTokens.UNK);
            }
            at_word_start = false;
            i = end;
        }

        // Step 2: BPE merge loop — greedily merge the highest-scoring adjacent pair
        while (tokens.items.len > 2) { // >2 because BOS is at position 0
            var best_score: f32 = -std.math.inf(f32);
            var best_idx: usize = 0;
            var best_id: u32 = 0;
            var found = false;

            // Find the highest-scoring adjacent merge (skip BOS at index 0)
            for (1..tokens.items.len - 1) |idx| {
                const left = tokens.items[idx];
                const right = tokens.items[idx + 1];

                // Concatenate left + right token strings and look up
                const left_str = self.vocab[left];
                const right_str = self.vocab[right];

                if (left_str.len + right_str.len > self.max_token_len) continue;

                var merge_buf: [256]u8 = undefined;
                if (left_str.len + right_str.len > merge_buf.len) continue;

                @memcpy(merge_buf[0..left_str.len], left_str);
                @memcpy(merge_buf[left_str.len..][0..right_str.len], right_str);
                const merged = merge_buf[0 .. left_str.len + right_str.len];

                if (self.token_to_id.get(merged)) |merged_id| {
                    const score = self.scores[merged_id];
                    if (score > best_score) {
                        best_score = score;
                        best_idx = idx;
                        best_id = merged_id;
                        found = true;
                    }
                }
            }

            if (!found) break;

            // Apply merge: replace tokens[best_idx] and tokens[best_idx+1] with merged token
            tokens.items[best_idx] = best_id;
            _ = tokens.orderedRemove(best_idx + 1);
        }

        return tokens.toOwnedSlice(allocator);
    }

    /// Decode token IDs to text.
    /// Skips BOS/EOS/PAD tokens. Replaces ▁ with space.
    /// Returns allocated string (caller owns memory).
    pub fn decode(self: *const Self, ids: []const u32) ![]u8 {
        const allocator = self.allocator;

        var result: std.ArrayListUnmanaged(u8) = .{};
        errdefer result.deinit(allocator);

        for (ids) |id| {
            // Skip special tokens
            if (id == SpecialTokens.BOS or id == SpecialTokens.EOS or id == SpecialTokens.PAD) continue;
            if (id >= self.vocab_size) continue;

            const token = self.vocab[id];

            // Replace ▁ (U+2581, 3 bytes: E2 96 81) with space
            var i: usize = 0;
            while (i < token.len) {
                if (i + 2 < token.len and token[i] == 0xE2 and token[i + 1] == 0x96 and token[i + 2] == 0x81) {
                    try result.append(allocator, ' ');
                    i += 3;
                } else {
                    try result.append(allocator, token[i]);
                    i += 1;
                }
            }
        }

        // Trim leading space (first token's ▁ becomes a leading space)
        const items = result.items;
        if (items.len > 0 and items[0] == ' ') {
            std.mem.copyForwards(u8, items[0 .. items.len - 1], items[1..]);
            result.items.len -= 1;
        }

        return result.toOwnedSlice(allocator);
    }

    pub fn deinit(self: *Self) void {
        for (self.vocab) |t| self.allocator.free(t);
        self.allocator.free(self.vocab);
        self.allocator.free(self.scores);
        self.token_to_id.deinit();
    }
};
