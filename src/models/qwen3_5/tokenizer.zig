// src/models/qwen3_5/tokenizer.zig
// Byte-level BPE tokenizer for Qwen3.5 (tiktoken-style, 248K vocab)
//
// Binary format (exported by export_qwen3_5.py):
//   u32: vocab_size
//   u32: num_merges
//   u32: max_token_len
//   For each token (ordered by ID):
//     u32: token_len
//     [token_len]u8: token bytes
//   For each merge (ordered by priority):
//     u32: pair_a_id
//     u32: pair_b_id
//     u32: merged_id

const std = @import("std");

pub const Tokenizer = struct {
    /// Token ID → byte string
    vocab: [][]const u8,
    /// Byte string → Token ID (for encoding)
    token_to_id: std.StringHashMapUnmanaged(u32),
    /// Merge rules: (pair_a, pair_b) → merged_id, ordered by priority
    merges: []Merge,
    /// Byte value → Token ID (handles GPT-2 byte-to-unicode encoding)
    byte_to_token: [256]u32,
    vocab_size: u32,
    allocator: std.mem.Allocator,

    pub const Merge = struct {
        pair_a: u32,
        pair_b: u32,
        merged: u32,
    };

    /// GPT-2 bytes_to_unicode: maps each byte to a Unicode codepoint.
    /// Printable bytes map to themselves; others map to U+0100..U+01FF.
    fn byteToUnicodeCodepoint(byte: u8) u21 {
        // Printable ASCII ranges that map to themselves:
        // 0x21-0x7E (! to ~), 0xA1-0xAC (¡ to ¬), 0xAE-0xFF (® to ÿ)
        return switch (byte) {
            0x21...0x7E, 0xA1...0xAC, 0xAE...0xFF => @intCast(byte),
            else => blk: {
                // Count how many "direct" bytes come before this one
                // Bytes 0x00-0x20, 0x7F-0xA0, 0xAD map to 0x100+n
                var n: u21 = 0;
                for (0..256) |b| {
                    switch (@as(u8, @intCast(b))) {
                        0x21...0x7E, 0xA1...0xAC, 0xAE...0xFF => {},
                        else => {
                            if (b == byte) break :blk 0x100 + n;
                            n += 1;
                        },
                    }
                }
                unreachable;
            },
        };
    }

    /// Encode a Unicode codepoint to UTF-8 bytes, return the slice.
    fn codepointToUtf8(cp: u21, buf: *[4]u8) []const u8 {
        const len = std.unicode.utf8Encode(cp, buf) catch unreachable;
        return buf[0..len];
    }

    pub fn initFromBytes(allocator: std.mem.Allocator, bytes: []const u8) !Tokenizer {
        var offset: usize = 0;

        const vocab_size = readU32(bytes, &offset);
        const num_merges = readU32(bytes, &offset);
        _ = readU32(bytes, &offset); // max_token_len (reserved)

        // Read vocabulary
        var vocab = try allocator.alloc([]const u8, vocab_size);
        errdefer {
            for (vocab[0..vocab_size]) |v| allocator.free(v);
            allocator.free(vocab);
        }

        var token_to_id: std.StringHashMapUnmanaged(u32) = .{};
        errdefer token_to_id.deinit(allocator);

        for (0..vocab_size) |id| {
            const token_len = readU32(bytes, &offset);
            const token_bytes = try allocator.alloc(u8, token_len);
            @memcpy(token_bytes, bytes[offset..][0..token_len]);
            offset += token_len;

            vocab[id] = token_bytes;
            try token_to_id.put(allocator, token_bytes, @intCast(id));
        }

        // Read merges
        var merges = try allocator.alloc(Merge, num_merges);
        for (0..num_merges) |i| {
            merges[i] = .{
                .pair_a = readU32(bytes, &offset),
                .pair_b = readU32(bytes, &offset),
                .merged = readU32(bytes, &offset),
            };
        }

        // Build byte → token ID lookup using GPT-2 bytes_to_unicode mapping
        var byte_to_token: [256]u32 = undefined;
        for (0..256) |b| {
            const cp = byteToUnicodeCodepoint(@intCast(b));
            var utf8_buf: [4]u8 = undefined;
            const utf8 = codepointToUtf8(cp, &utf8_buf);
            if (token_to_id.get(utf8)) |id| {
                byte_to_token[b] = id;
            } else {
                // Fallback: try raw byte
                if (token_to_id.get(&[_]u8{@intCast(b)})) |id| {
                    byte_to_token[b] = id;
                } else {
                    byte_to_token[b] = @intCast(b); // Last resort
                }
            }
        }

        return .{
            .vocab = vocab,
            .token_to_id = token_to_id,
            .merges = merges,
            .byte_to_token = byte_to_token,
            .vocab_size = vocab_size,
            .allocator = allocator,
        };
    }

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !Tokenizer {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const data = try file.readToEndAlloc(allocator, 256 * 1024 * 1024);
        defer allocator.free(data);
        return initFromBytes(allocator, data);
    }

    /// Encode text into token IDs using byte-level BPE.
    /// Handles special tokens (e.g. <|im_start|>) by matching them before BPE.
    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        if (text.len == 0) {
            return try allocator.alloc(u32, 0);
        }

        var ids = std.ArrayListUnmanaged(u32){};
        errdefer ids.deinit(allocator);

        var pos: usize = 0;
        while (pos < text.len) {
            // Check for special token at current position (patterns like <|...|>)
            if (text[pos] == '<' and pos + 2 < text.len and text[pos + 1] == '|') {
                if (self.matchSpecialToken(text[pos..])) |match| {
                    try ids.append(allocator, match.id);
                    pos += match.len;
                    continue;
                }
            }

            // Find the end of the current non-special segment
            var seg_end = pos + 1;
            while (seg_end < text.len) {
                if (text[seg_end] == '<' and seg_end + 2 < text.len and text[seg_end + 1] == '|') {
                    if (self.matchSpecialToken(text[seg_end..])) |_| break;
                }
                seg_end += 1;
            }

            // BPE-encode the non-special segment
            const segment = text[pos..seg_end];
            const seg_ids = try self.encodeBpe(allocator, segment);
            defer allocator.free(seg_ids);
            try ids.appendSlice(allocator, seg_ids);
            pos = seg_end;
        }

        return ids.toOwnedSlice(allocator);
    }

    const SpecialMatch = struct { id: u32, len: usize };

    /// Try to match a special token at the start of text. Returns ID and byte length if found.
    fn matchSpecialToken(self: *const Tokenizer, text: []const u8) ?SpecialMatch {
        // Find the closing |> to extract the candidate
        if (text.len < 4) return null; // minimum: <|x|>
        var end: usize = 2;
        while (end + 1 < text.len and end < 64) : (end += 1) { // cap at 64 chars
            if (text[end] == '|' and text[end + 1] == '>') {
                const token_str = text[0 .. end + 2]; // <|...|>
                if (self.token_to_id.get(token_str)) |id| {
                    return .{ .id = id, .len = token_str.len };
                }
                return null; // Found |> but not in vocab
            }
        }
        return null;
    }

    /// Core BPE encoding for a text segment (no special tokens).
    fn encodeBpe(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        if (text.len == 0) {
            return try allocator.alloc(u32, 0);
        }

        // Step 1: Initialize with byte-level tokens (using GPT-2 byte mapping)
        var ids_list = std.ArrayListUnmanaged(u32){};
        errdefer ids_list.deinit(allocator);

        for (text) |byte| {
            try ids_list.append(allocator, self.byte_to_token[byte]);
        }

        // Step 2: Apply BPE merges greedily
        for (self.merges) |merge| {
            var i: usize = 0;
            while (i + 1 < ids_list.items.len) {
                if (ids_list.items[i] == merge.pair_a and ids_list.items[i + 1] == merge.pair_b) {
                    ids_list.items[i] = merge.merged;
                    std.mem.copyForwards(u32, ids_list.items[i + 1 ..], ids_list.items[i + 2 ..]);
                    ids_list.items.len -= 1;
                } else {
                    i += 1;
                }
            }
        }

        return ids_list.toOwnedSlice(allocator);
    }

    /// Decode token IDs back to raw bytes, reversing GPT-2 byte encoding.
    pub fn decode(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        var result = std.ArrayListUnmanaged(u8){};
        errdefer result.deinit(allocator);

        for (ids) |id| {
            if (id < self.vocab_size) {
                const token = self.vocab[id];
                // Decode UTF-8 codepoints and reverse-map GPT-2 byte encoding
                var i: usize = 0;
                while (i < token.len) {
                    const cp_len = std.unicode.utf8ByteSequenceLength(token[i]) catch {
                        try result.append(allocator, token[i]);
                        i += 1;
                        continue;
                    };
                    if (i + cp_len > token.len) {
                        try result.append(allocator, token[i]);
                        i += 1;
                        continue;
                    }
                    const cp = std.unicode.utf8Decode(token[i..][0..cp_len]) catch {
                        try result.append(allocator, token[i]);
                        i += 1;
                        continue;
                    };
                    if (unicodeToByte(cp)) |byte| {
                        try result.append(allocator, byte);
                    } else {
                        // Not a GPT-2 mapped codepoint, output raw UTF-8
                        try result.appendSlice(allocator, token[i..][0..cp_len]);
                    }
                    i += cp_len;
                }
            }
        }

        return result.toOwnedSlice(allocator);
    }

    /// Reverse GPT-2 bytes_to_unicode: map a Unicode codepoint back to a byte value.
    fn unicodeToByte(cp: u21) ?u8 {
        // Direct mapping: printable ASCII and high bytes map to themselves
        if ((cp >= 0x21 and cp <= 0x7E) or (cp >= 0xA1 and cp <= 0xAC) or (cp >= 0xAE and cp <= 0xFF)) {
            return @intCast(cp);
        }
        // Mapped range: U+0100..U+0143 map to the non-printable bytes
        if (cp >= 0x100 and cp <= 0x143) {
            const n: u8 = @intCast(cp - 0x100);
            // Reconstruct: the n-th non-printable byte
            var count: u8 = 0;
            for (0..256) |b| {
                switch (@as(u8, @intCast(b))) {
                    0x21...0x7E, 0xA1...0xAC, 0xAE...0xFF => {},
                    else => {
                        if (count == n) return @intCast(b);
                        count += 1;
                    },
                }
            }
        }
        return null;
    }

    /// Format a chat prompt with Qwen3.5 chat template.
    pub fn formatChatPrompt(allocator: std.mem.Allocator, system_msg: ?[]const u8, user_msg: []const u8) ![]u8 {
        var result = std.ArrayListUnmanaged(u8){};
        errdefer result.deinit(allocator);

        const writer = result.writer(allocator);

        if (system_msg) |sys| {
            try writer.writeAll("<|im_start|>system\n");
            try writer.writeAll(sys);
            try writer.writeAll("<|im_end|>\n");
        }

        try writer.writeAll("<|im_start|>user\n");
        try writer.writeAll(user_msg);
        try writer.writeAll("<|im_end|>\n");
        try writer.writeAll("<|im_start|>assistant\n");

        return result.toOwnedSlice(allocator);
    }

    pub fn deinit(self: *Tokenizer) void {
        for (self.vocab) |v| self.allocator.free(v);
        self.allocator.free(self.vocab);
        self.token_to_id.deinit(self.allocator);
        self.allocator.free(self.merges);
    }

    fn readU32(bytes: []const u8, offset: *usize) u32 {
        const val = std.mem.readInt(u32, bytes[offset.*..][0..4], .little);
        offset.* += 4;
        return val;
    }
};
