// src/models/sentence_transformer/tokenizer.zig
// WordPiece Tokenizer for BERT-based models (all-MiniLM-L6-v2)
//
// Algorithm: BERT-style WordPiece
//   1. Lowercase input
//   2. Split on whitespace and punctuation
//   3. Greedy longest-match sub-word tokenization with ## continuation prefix
//   4. Wrap with [CLS] ... [SEP]

const std = @import("std");

/// Special token IDs (BERT vocabulary)
pub const SpecialTokens = struct {
    pub const PAD: u32 = 0; // [PAD]
    pub const UNK: u32 = 100; // [UNK]
    pub const CLS: u32 = 101; // [CLS]
    pub const SEP: u32 = 102; // [SEP]
};

/// Encoded output from tokenizer
pub const TokenizerOutput = struct {
    input_ids: []u32,
    attention_mask: []u32,
    token_type_ids: []u32,

    pub fn deinit(self: *TokenizerOutput, allocator: std.mem.Allocator) void {
        allocator.free(self.input_ids);
        allocator.free(self.attention_mask);
        allocator.free(self.token_type_ids);
    }
};

/// WordPiece Tokenizer for BERT models
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMap(u32),
    id_to_token: std.AutoHashMap(u32, []const u8),
    vocab_size: u32,
    max_seq_len: usize,

    const Self = @This();
    const CONTINUATION_PREFIX = "##";
    const MAX_WORD_LEN: usize = 200; // Max chars per word for tokenization

    /// Initialize tokenizer from vocabulary file path
    pub fn init(allocator: std.mem.Allocator, vocab_path: []const u8) !Self {
        const file = std.fs.cwd().openFile(vocab_path, .{}) catch |err| {
            std.debug.print("Error opening vocab file '{s}': {}\n", .{ vocab_path, err });
            return err;
        };
        defer file.close();

        const content = try file.readToEndAlloc(allocator, 100 * 1024 * 1024);
        defer allocator.free(content);

        return Self.initFromString(allocator, content);
    }

    /// Initialize tokenizer from vocabulary string content
    pub fn initFromString(allocator: std.mem.Allocator, vocab_content: []const u8) !Self {
        var vocab = std.StringHashMap(u32).init(allocator);
        errdefer vocab.deinit();

        var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
        errdefer id_to_token.deinit();

        var token_id: u32 = 0;
        var iter = std.mem.splitScalar(u8, vocab_content, '\n');

        while (iter.next()) |line| {
            if (line.len == 0 and iter.peek() == null) {
                // Skip trailing empty line
                break;
            }

            const unescaped = try unescapeToken(allocator, line);
            try vocab.put(unescaped, token_id);
            try id_to_token.put(token_id, unescaped);
            token_id += 1;
        }

        return Self{
            .allocator = allocator,
            .vocab = vocab,
            .id_to_token = id_to_token,
            .vocab_size = token_id,
            .max_seq_len = 512,
        };
    }

    /// Free tokenizer memory
    pub fn deinit(self: *Self) void {
        var it = self.id_to_token.valueIterator();
        while (it.next()) |token| {
            self.allocator.free(token.*);
        }
        self.vocab.deinit();
        self.id_to_token.deinit();
    }

    /// Encode text to token IDs with attention mask and token type IDs
    /// Adds [CLS] at start and [SEP] at end
    pub fn encode(self: *Self, text: []const u8) !TokenizerOutput {
        var ids: std.ArrayList(u32) = .{};
        errdefer ids.deinit(self.allocator);

        // Add [CLS]
        try ids.append(self.allocator, SpecialTokens.CLS);

        // Tokenize the text
        try self.tokenizeInto(text, &ids);

        // Add [SEP]
        try ids.append(self.allocator, SpecialTokens.SEP);

        // Truncate if needed (keep [CLS] at start and [SEP] at end)
        if (ids.items.len > self.max_seq_len) {
            ids.items[self.max_seq_len - 1] = SpecialTokens.SEP;
            ids.shrinkRetainingCapacity(self.max_seq_len);
        }

        const seq_len = ids.items.len;

        // Build attention mask (all 1s for non-padding)
        const attention_mask = try self.allocator.alloc(u32, seq_len);
        errdefer self.allocator.free(attention_mask);
        @memset(attention_mask, 1);

        // Build token type IDs (all 0s for single sequence)
        const token_type_ids = try self.allocator.alloc(u32, seq_len);
        errdefer self.allocator.free(token_type_ids);
        @memset(token_type_ids, 0);

        return TokenizerOutput{
            .input_ids = try ids.toOwnedSlice(self.allocator),
            .attention_mask = attention_mask,
            .token_type_ids = token_type_ids,
        };
    }

    /// Tokenize text and append token IDs to the list
    fn tokenizeInto(self: *Self, text: []const u8, ids: *std.ArrayList(u32)) !void {
        // Lowercase the input
        var lower_buf = try self.allocator.alloc(u8, text.len);
        defer self.allocator.free(lower_buf);
        for (text, 0..) |c, i| {
            lower_buf[i] = if (c >= 'A' and c <= 'Z') c + 32 else c;
        }

        // Split into words on whitespace and punctuation boundaries
        var start: usize = 0;
        while (start < lower_buf.len) {
            // Skip whitespace
            if (isWhitespace(lower_buf[start])) {
                start += 1;
                continue;
            }

            // Check if current char is punctuation
            if (isPunctuation(lower_buf[start])) {
                // Punctuation is its own token
                try self.tokenizeWord(lower_buf[start .. start + 1], ids);
                start += 1;
                continue;
            }

            // Find end of word (non-whitespace, non-punctuation)
            var end = start + 1;
            while (end < lower_buf.len and !isWhitespace(lower_buf[end]) and !isPunctuation(lower_buf[end])) {
                end += 1;
            }

            try self.tokenizeWord(lower_buf[start..end], ids);
            start = end;
        }
    }

    /// Tokenize a single word using WordPiece greedy longest-match
    fn tokenizeWord(self: *Self, word: []const u8, ids: *std.ArrayList(u32)) !void {
        if (word.len == 0) return;

        // Try whole word first
        if (self.vocab.get(word)) |id| {
            try ids.append(self.allocator, id);
            return;
        }

        // WordPiece sub-word tokenization
        var start: usize = 0;
        var is_first = true;

        // Buffer for building ##prefix tokens
        var sub_buf: [MAX_WORD_LEN + 2]u8 = undefined;

        while (start < word.len) {
            var end = word.len;
            var found = false;

            while (end > start) {
                const substr = word[start..end];

                if (is_first) {
                    // First piece: try as-is
                    if (self.vocab.get(substr)) |id| {
                        try ids.append(self.allocator, id);
                        start = end;
                        is_first = false;
                        found = true;
                        break;
                    }
                } else {
                    // Continuation: try with ## prefix
                    if (substr.len + 2 <= sub_buf.len) {
                        sub_buf[0] = '#';
                        sub_buf[1] = '#';
                        @memcpy(sub_buf[2 .. 2 + substr.len], substr);
                        const prefixed = sub_buf[0 .. 2 + substr.len];

                        if (self.vocab.get(prefixed)) |id| {
                            try ids.append(self.allocator, id);
                            start = end;
                            found = true;
                            break;
                        }
                    }
                }

                end -= 1;
            }

            if (!found) {
                // Character not in vocab — emit [UNK] for entire remaining word
                try ids.append(self.allocator, SpecialTokens.UNK);
                break;
            }
        }
    }

    /// Get the token string for a specific ID
    pub fn getToken(self: *Self, id: u32) ?[]const u8 {
        return self.id_to_token.get(id);
    }
};

fn isWhitespace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r';
}

fn isPunctuation(c: u8) bool {
    return switch (c) {
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/' => true,
        ':', ';', '<', '=', '>', '?', '@' => true,
        '[', '\\', ']', '^', '_', '`' => true,
        '{', '|', '}', '~' => true,
        else => false,
    };
}

/// Unescape special characters in token (from vocab file)
fn unescapeToken(allocator: std.mem.Allocator, token: []const u8) ![]const u8 {
    var result: std.ArrayList(u8) = .{};
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i < token.len) {
        if (token[i] == '\\' and i + 1 < token.len) {
            switch (token[i + 1]) {
                'n' => {
                    try result.append(allocator, '\n');
                    i += 2;
                },
                't' => {
                    try result.append(allocator, '\t');
                    i += 2;
                },
                '\\' => {
                    try result.append(allocator, '\\');
                    i += 2;
                },
                else => {
                    try result.append(allocator, token[i]);
                    i += 1;
                },
            }
        } else {
            try result.append(allocator, token[i]);
            i += 1;
        }
    }

    return result.toOwnedSlice(allocator);
}
