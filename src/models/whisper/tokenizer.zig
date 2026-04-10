// Whisper Tokenizer - Pure Zig
//
// Decodes Whisper token IDs to text using a vocabulary file.
// The vocabulary is exported from the Whisper tokenizer using export_whisper_vocab.py.
//
// Vocabulary format (binary):
//   - 4 bytes: number of tokens (u32 little-endian)
//   - For each token:
//     - 2 bytes: token length (u16 little-endian)
//     - N bytes: token string (UTF-8)

const std = @import("std");

/// Special token IDs in Whisper
pub const SpecialTokens = struct {
    pub const EOT: u32 = 50257; // <|endoftext|>
    pub const SOT: u32 = 50258; // <|startoftranscript|>
    pub const TRANSCRIBE: u32 = 50359;
    pub const TRANSLATE: u32 = 50358;
    pub const NO_SPEECH: u32 = 50362;
    pub const NO_TIMESTAMPS: u32 = 50363;
    pub const TIMESTAMP_BEGIN: u32 = 50364;

    // Language tokens range: 50259-50357
    pub const LANG_START: u32 = 50259;
    pub const LANG_END: u32 = 50357;
};

/// Whisper tokenizer for decoding token IDs to text
pub const WhisperTokenizer = struct {
    vocab: [][]const u8,
    vocab_size: u32,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Load tokenizer from binary vocabulary file
    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !Self {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        // Read the entire file into memory
        const stat = try file.stat();
        const file_size = stat.size;
        const file_data = try allocator.alloc(u8, file_size);
        defer allocator.free(file_data);

        const bytes_read = try file.readAll(file_data);
        if (bytes_read != file_size) {
            return error.UnexpectedEndOfFile;
        }

        // Parse the binary format
        if (file_data.len < 4) return error.InvalidFormat;

        // Read vocab size (first 4 bytes, little-endian)
        const vocab_size = std.mem.readInt(u32, file_data[0..4], .little);

        // Allocate vocab array
        var vocab = try allocator.alloc([]const u8, vocab_size);
        errdefer {
            for (vocab) |token| {
                allocator.free(token);
            }
            allocator.free(vocab);
        }

        // Read each token
        var pos: usize = 4;
        for (0..vocab_size) |i| {
            if (pos + 2 > file_data.len) return error.UnexpectedEndOfFile;

            const token_len = std.mem.readInt(u16, file_data[pos..][0..2], .little);
            pos += 2;

            if (pos + token_len > file_data.len) return error.UnexpectedEndOfFile;

            const token = try allocator.alloc(u8, token_len);
            errdefer allocator.free(token);

            @memcpy(token, file_data[pos..][0..token_len]);
            pos += token_len;

            vocab[i] = token;
        }

        return Self{
            .vocab = vocab,
            .vocab_size = vocab_size,
            .allocator = allocator,
        };
    }

    /// Deinitialize the tokenizer
    pub fn deinit(self: *Self) void {
        for (self.vocab) |token| {
            self.allocator.free(token);
        }
        self.allocator.free(self.vocab);
    }

    /// Decode a single token ID to text
    pub fn decode(self: *const Self, token_id: u32) ?[]const u8 {
        if (token_id >= self.vocab_size) {
            return null;
        }
        return self.vocab[token_id];
    }

    /// Check if a token is a special token (should be skipped in output)
    pub fn isSpecialToken(token_id: u32) bool {
        // EOT, SOT, and other control tokens
        if (token_id >= SpecialTokens.EOT) {
            return true;
        }
        return false;
    }

    /// Check if a token is a timestamp token
    pub fn isTimestampToken(token_id: u32) bool {
        return token_id >= SpecialTokens.TIMESTAMP_BEGIN;
    }

    /// Decode a sequence of tokens to text
    /// Skips special tokens and optionally includes timestamps
    pub fn decodeTokens(self: *const Self, allocator: std.mem.Allocator, tokens: []const u32, include_timestamps: bool) ![]u8 {
        var result: std.ArrayListUnmanaged(u8) = .{};
        errdefer result.deinit(allocator);

        for (tokens) |token_id| {
            // Skip special tokens (except timestamps if requested)
            if (isSpecialToken(token_id)) {
                if (include_timestamps and isTimestampToken(token_id)) {
                    if (self.decode(token_id)) |text| {
                        try result.appendSlice(allocator, text);
                    }
                }
                continue;
            }

            // Decode regular token
            if (self.decode(token_id)) |text| {
                // Unescape the token text
                try unescapeToken(allocator, &result, text);
            }
        }

        return result.toOwnedSlice(allocator);
    }

    /// Decode tokens to text, also returning the raw text (with special tokens)
    pub fn decodeTokensRaw(self: *const Self, allocator: std.mem.Allocator, tokens: []const u32) ![]u8 {
        var result: std.ArrayListUnmanaged(u8) = .{};
        errdefer result.deinit(allocator);

        for (tokens) |token_id| {
            if (self.decode(token_id)) |text| {
                try result.appendSlice(allocator, text);
            }
        }

        return result.toOwnedSlice(allocator);
    }
};

/// Unescape a token string (handle \n, \t, \r, \\)
fn unescapeToken(allocator: std.mem.Allocator, result: *std.ArrayListUnmanaged(u8), text: []const u8) !void {
    var i: usize = 0;
    while (i < text.len) {
        if (text[i] == '\\' and i + 1 < text.len) {
            switch (text[i + 1]) {
                'n' => {
                    try result.append(allocator, '\n');
                    i += 2;
                    continue;
                },
                't' => {
                    try result.append(allocator, '\t');
                    i += 2;
                    continue;
                },
                'r' => {
                    try result.append(allocator, '\r');
                    i += 2;
                    continue;
                },
                '\\' => {
                    try result.append(allocator, '\\');
                    i += 2;
                    continue;
                },
                else => {},
            }
        }
        try result.append(allocator, text[i]);
        i += 1;
    }
}

test "tokenizer basic" {
    // This test requires the vocab file to exist
    const allocator = std.testing.allocator;

    var tokenizer = WhisperTokenizer.loadFromFile(allocator, "models/whisper_vocab.bin") catch |err| {
        std.debug.print("Skipping test: vocab file not found: {}\n", .{err});
        return;
    };
    defer tokenizer.deinit();

    // Test basic decoding
    // Token 0 should be "!" (first printable ASCII after space handling)
    const token_0 = tokenizer.decode(0);
    try std.testing.expect(token_0 != null);

    // Test special token detection
    try std.testing.expect(WhisperTokenizer.isSpecialToken(50257)); // EOT
    try std.testing.expect(WhisperTokenizer.isSpecialToken(50258)); // SOT
    try std.testing.expect(!WhisperTokenizer.isSpecialToken(0)); // Regular token
    try std.testing.expect(!WhisperTokenizer.isSpecialToken(1000)); // Regular token
}
