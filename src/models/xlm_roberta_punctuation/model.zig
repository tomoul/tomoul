// src/models/xlm_roberta/model.zig
// XLM-RoBERTa Transformer Encoder with SentencePiece Tokenizer (Phase 6)
//
// Supports models:
//   - oliverguhr/fullstop-punctuation-multilang-large (24 layers, 1024 hidden)
//   - kredor/punctuate-all (12 layers, 768 hidden)
//   - oliverguhr/fullstop-punctuation-multilingual-sonar-base (12 layers, 768 hidden)
//
// Architecture: XLM-RoBERTa encoder + token classification head
// Tokenizer: SentencePiece (250k vocab) - integrated in this file

const std = @import("std");
const tensor_mod = @import("tensor.zig");
const Tensor = tensor_mod.Tensor;
const ops = @import("ops.zig");
const loader_mod = @import("loader.zig");
const ModelLoader = loader_mod.ModelLoader;
const LoadError = loader_mod.LoadError;
const QuantFormat = loader_mod.QuantFormat;

// Transformer/attention modules (supports F32, Q8, Q4, Q8_K via comptime generics)
const transformer = @import("transformer.zig");
const attention = @import("attention.zig");
const TransformerBlockWeights = transformer.TransformerBlockWeightsF32;
const TransformerBlockWeightsQ8 = transformer.TransformerBlockWeightsQ8;
const TransformerBlockWeightsQ4 = transformer.TransformerBlockWeightsQ4;
const TransformerBlockWeightsQ8K = transformer.TransformerBlockWeightsQ8K;
const TransformerConfig = transformer.TransformerConfig;
const AttentionWeights = attention.AttentionWeightsF32;
const AttentionWeightsQ8 = attention.AttentionWeightsQ8;
const AttentionWeightsQ4 = attention.AttentionWeightsQ4;
const AttentionWeightsQ8K = attention.AttentionWeightsQ8K;

// Quantization support
const quant = @import("quantization.zig");
const QuantizedTensorQ8 = quant.QuantizedTensorQ8;
const QuantizedTensorQ4 = quant.QuantizedTensorQ4;
const QuantizedTensorQ8K = quant.QuantizedTensorQ8K;

// =============================================================================
// Tokenizer (SentencePiece for XLM-RoBERTa)
// =============================================================================

/// Special token IDs (XLM-RoBERTa vocabulary)
pub const SpecialTokens = struct {
    pub const BOS: u32 = 0; // <s> (beginning of sequence)
    pub const PAD: u32 = 1; // <pad>
    pub const EOS: u32 = 2; // </s> (end of sequence)
    pub const UNK: u32 = 3; // <unk>
    // Note: <mask> is at position 250001 in XLM-RoBERTa vocab
};

/// Encoded output from tokenizer
pub const EncodedOutput = struct {
    ids: []u32,
    tokens: [][]const u8,
};

/// SentencePiece Tokenizer for XLM-RoBERTa models
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMap(u32),
    id_to_token: std.AutoHashMap(u32, []const u8),
    vocab_size: u32,

    const Self = @This();

    // SentencePiece word boundary marker (U+2581 = "▁")
    const WORD_BOUNDARY = "▁";
    const WORD_BOUNDARY_BYTES = [_]u8{ 0xE2, 0x96, 0x81 }; // UTF-8 encoding

    /// Initialize tokenizer from vocabulary file
    pub fn init(allocator: std.mem.Allocator, vocab_path: []const u8) !Self {
        const file = std.fs.cwd().openFile(vocab_path, .{}) catch |err| {
            std.debug.print("Error opening vocab file '{s}': {}\n", .{ vocab_path, err });
            return err;
        };
        defer file.close();

        const content = try file.readToEndAlloc(allocator, 100 * 1024 * 1024); // 100MB max
        defer allocator.free(content);

        return Self.initFromString(allocator, content);
    }

    /// Initialize tokenizer from memory (vocab as string)
    pub fn initFromString(allocator: std.mem.Allocator, vocab_content: []const u8) !Self {
        var vocab = std.StringHashMap(u32).init(allocator);
        errdefer vocab.deinit();

        var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
        errdefer id_to_token.deinit();

        var token_id: u32 = 0;
        var iter = std.mem.splitScalar(u8, vocab_content, '\n');

        while (iter.next()) |line| {
            if (line.len == 0) {
                token_id += 1;
                continue;
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

    /// Encode text to token IDs with token strings
    /// Adds <s> at start and </s> at end
    pub fn encode(self: *Self, text: []const u8) !EncodedOutput {
        var ids: std.ArrayList(u32) = .{};
        errdefer ids.deinit(self.allocator);

        var tokens: std.ArrayList([]const u8) = .{};
        errdefer {
            for (tokens.items) |t| self.allocator.free(t);
            tokens.deinit(self.allocator);
        }

        // Add <s> token (BOS)
        try ids.append(self.allocator, SpecialTokens.BOS);
        try tokens.append(self.allocator, try self.allocator.dupe(u8, "<s>"));

        // Tokenize the text
        try self.tokenizeInto(text, &ids, &tokens);

        // Add </s> token (EOS)
        try ids.append(self.allocator, SpecialTokens.EOS);
        try tokens.append(self.allocator, try self.allocator.dupe(u8, "</s>"));

        return EncodedOutput{
            .ids = try ids.toOwnedSlice(self.allocator),
            .tokens = try tokens.toOwnedSlice(self.allocator),
        };
    }

    /// Tokenize text and append to existing lists
    fn tokenizeInto(self: *Self, text: []const u8, ids: *std.ArrayList(u32), tokens: *std.ArrayList([]const u8)) !void {
        // Convert to lowercase
        var lower_text = try self.allocator.alloc(u8, text.len);
        defer self.allocator.free(lower_text);
        for (text, 0..) |c, i| {
            lower_text[i] = if (c >= 'A' and c <= 'Z') c + 32 else c;
        }

        // Split on whitespace
        var word_iter = std.mem.tokenizeAny(u8, lower_text, " \t\n\r");

        while (word_iter.next()) |word| {
            try self.tokenizeWord(word, ids, tokens);
        }
    }

    /// Tokenize a single word using greedy longest-match
    fn tokenizeWord(self: *Self, word: []const u8, ids: *std.ArrayList(u32), tokens: *std.ArrayList([]const u8)) !void {
        // Build word with boundary marker
        var full_word: std.ArrayList(u8) = .{};
        defer full_word.deinit(self.allocator);

        try full_word.appendSlice(self.allocator, &WORD_BOUNDARY_BYTES);
        try full_word.appendSlice(self.allocator, word);

        const word_with_marker = full_word.items;

        // Try whole word first
        if (self.vocab.get(word_with_marker)) |id| {
            try ids.append(self.allocator, id);
            try tokens.append(self.allocator, try self.allocator.dupe(u8, word_with_marker));
            return;
        }

        // Greedy longest-match tokenization
        var start: usize = 0;
        while (start < word_with_marker.len) {
            var end = word_with_marker.len;
            var found = false;

            while (end > start) {
                const subword = word_with_marker[start..end];

                if (self.vocab.get(subword)) |id| {
                    try ids.append(self.allocator, id);
                    try tokens.append(self.allocator, try self.allocator.dupe(u8, subword));
                    start = end;
                    found = true;
                    break;
                }

                end = prevCharBoundary(word_with_marker, end);
            }

            if (!found) {
                try ids.append(self.allocator, SpecialTokens.UNK);
                try tokens.append(self.allocator, try self.allocator.dupe(u8, "<unk>"));
                start = nextCharBoundary(word_with_marker, start);
            }
        }
    }

    /// Decode token IDs back to text
    pub fn decode(self: *Self, token_ids: []const u32) ![]u8 {
        var result: std.ArrayList(u8) = .{};
        errdefer result.deinit(self.allocator);

        for (token_ids) |id| {
            if (id == SpecialTokens.BOS or id == SpecialTokens.EOS or id == SpecialTokens.PAD) {
                continue;
            }

            if (self.id_to_token.get(id)) |token| {
                if (std.mem.startsWith(u8, token, &WORD_BOUNDARY_BYTES)) {
                    if (result.items.len > 0) {
                        try result.append(self.allocator, ' ');
                    }
                    try result.appendSlice(self.allocator, token[WORD_BOUNDARY_BYTES.len..]);
                } else {
                    try result.appendSlice(self.allocator, token);
                }
            } else {
                try result.appendSlice(self.allocator, "<unk>");
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    /// Get the token string for a specific ID
    pub fn getToken(self: *Self, id: u32) ?[]const u8 {
        return self.id_to_token.get(id);
    }
};

/// Find previous UTF-8 character boundary
fn prevCharBoundary(text: []const u8, pos: usize) usize {
    if (pos == 0) return 0;
    var i = pos - 1;
    while (i > 0 and (text[i] & 0xC0) == 0x80) {
        i -= 1;
    }
    return i;
}

/// Find next UTF-8 character boundary
fn nextCharBoundary(text: []const u8, pos: usize) usize {
    if (pos >= text.len) return text.len;
    var i = pos + 1;
    while (i < text.len and (text[i] & 0xC0) == 0x80) {
        i += 1;
    }
    return i;
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

// =============================================================================
// Model (XLM-RoBERTa Token Classification)
// =============================================================================

/// Punctuation labels for token classification
pub const PunctuationLabel = enum(u8) {
    O = 0, // No punctuation
    COMMA = 1, // ,
    PERIOD = 2, // .
    QUESTION = 3, // ?
    PERIOD_U = 4, // . + uppercase next
    COMMA_U = 5, // , + uppercase next

    pub fn toChar(self: PunctuationLabel) ?u8 {
        return switch (self) {
            .O => null,
            .COMMA, .COMMA_U => ',',
            .PERIOD, .PERIOD_U => '.',
            .QUESTION => '?',
        };
    }

    pub fn shouldCapitalizeNext(self: PunctuationLabel) bool {
        return switch (self) {
            .PERIOD_U, .COMMA_U, .QUESTION => true,
            else => false,
        };
    }
};

/// XLM-RoBERTa configuration
pub const XLMRobertaConfig = struct {
    vocab_size: usize = 250002, // SentencePiece vocabulary
    hidden_dim: usize = 768, // Hidden dimension (768 for base, 1024 for large)
    intermediate_dim: usize = 3072, // FFN intermediate dimension (4x hidden)
    num_layers: usize = 12, // Number of transformer blocks (12 for base, 24 for large)
    num_heads: usize = 12, // Attention heads (12 for base, 16 for large)
    max_seq_len: usize = 514, // Maximum sequence length
    num_labels: usize = 6, // Classification labels
    layer_norm_eps: f32 = 1e-5, // RoBERTa uses 1e-5

    pub fn getTransformerConfig(self: XLMRobertaConfig) TransformerConfig {
        return TransformerConfig{
            .hidden_dim = self.hidden_dim,
            .intermediate_dim = self.intermediate_dim,
            .num_heads = self.num_heads,
            .layer_norm_eps = self.layer_norm_eps,
        };
    }
};

/// Configuration presets
pub const CONFIG_BASE = XLMRobertaConfig{
    .hidden_dim = 768,
    .intermediate_dim = 3072,
    .num_layers = 12,
    .num_heads = 12,
};

pub const CONFIG_LARGE = XLMRobertaConfig{
    .hidden_dim = 1024,
    .intermediate_dim = 4096,
    .num_layers = 24,
    .num_heads = 16,
};

/// Weight storage that can be float32, Q8_0, Q4_0, or Q8_K quantized
pub const WeightStorage = union(enum) {
    f32: struct {
        blocks: []TransformerBlockWeights,
        classifier_weight: Tensor,
    },
    q8: struct {
        blocks: []TransformerBlockWeightsQ8,
        classifier_weight: QuantizedTensorQ8,
    },
    q4: struct {
        blocks: []TransformerBlockWeightsQ4,
        classifier_weight: QuantizedTensorQ4,
    },
    q8k: struct {
        blocks: []TransformerBlockWeightsQ8K,
        classifier_weight: QuantizedTensorQ8K,
    },

    pub fn deinit(self: *WeightStorage, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .f32 => |*f| {
                for (f.blocks) |*block| block.deinit();
                allocator.free(f.blocks);
                f.classifier_weight.deinit();
            },
            .q8 => |*q| {
                for (q.blocks) |*block| block.deinit();
                allocator.free(q.blocks);
                q.classifier_weight.deinit();
            },
            .q4 => |*q| {
                for (q.blocks) |*block| block.deinit();
                allocator.free(q.blocks);
                q.classifier_weight.deinit();
            },
            .q8k => |*q| {
                for (q.blocks) |*block| block.deinit();
                allocator.free(q.blocks);
                q.classifier_weight.deinit();
            },
        }
    }
};

/// XLM-RoBERTa Token Classification Model
/// Supports both float32 and Q8_0 quantized weights - auto-detected at load time.
/// Used for punctuation restoration and other sequence labeling tasks.
pub const XLMRobertaModel = struct {
    allocator: std.mem.Allocator,
    config: XLMRobertaConfig,
    quant_format: QuantFormat,

    // Embeddings (always float32 - used for lookups, not matmul)
    word_embeddings: Tensor, // [vocab_size, hidden_dim]
    position_embeddings: Tensor, // [max_seq_len, hidden_dim]
    token_type_embeddings: Tensor, // [1, hidden_dim] (RoBERTa only uses 1)
    embed_ln_gamma: Tensor, // [hidden_dim]
    embed_ln_beta: Tensor, // [hidden_dim]

    // Transformer blocks + classifier (float32 or Q8_0)
    weights: WeightStorage,

    // Classification bias (always float32)
    classifier_bias: Tensor, // [num_labels]

    const Self = @This();

    /// Load XLM-RoBERTa model from .tl file
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
        var loader = try ModelLoader.init(allocator, model_path);
        defer loader.deinit();
        return Self.initFromLoader(allocator, &loader, null);
    }

    /// Load XLM-RoBERTa model with specific configuration
    pub fn initWithConfig(allocator: std.mem.Allocator, model_path: []const u8, config: XLMRobertaConfig) !Self {
        var loader = try ModelLoader.init(allocator, model_path);
        defer loader.deinit();
        return Self.initFromLoader(allocator, &loader, config);
    }

    /// Load XLM-RoBERTa model from embedded bytes (for Wasm)
    pub fn initFromBytes(allocator: std.mem.Allocator, model_bytes: []const u8) !Self {
        var loader = try ModelLoader.initFromBytes(allocator, model_bytes);
        defer loader.deinit();
        return Self.initFromLoader(allocator, &loader, null);
    }

    /// Internal: Initialize from a ModelLoader
    /// Auto-detects quantization format (float32, Q8_0, Q4_0, or Q8_K) from file header
    fn initFromLoader(allocator: std.mem.Allocator, loader: *ModelLoader, config_opt: ?XLMRobertaConfig) !Self {
        // Auto-detect config from model weights if not provided
        const config = config_opt orelse try detectConfig(loader);
        const quant_format = loader.quant_format;

        // Load embeddings - always as float32 (for lookups)
        // For quantized files, getTensorDequantized handles conversion
        const is_quantized = quant_format == .q8_0 or quant_format == .q4_0 or quant_format == .q8_k;

        var word_embeddings = if (is_quantized)
            try loader.getTensorDequantized("roberta.embeddings.word_embeddings.weight")
        else
            try loader.getTensor("roberta.embeddings.word_embeddings.weight");
        errdefer word_embeddings.deinit();

        var position_embeddings = if (is_quantized)
            try loader.getTensorDequantized("roberta.embeddings.position_embeddings.weight")
        else
            try loader.getTensor("roberta.embeddings.position_embeddings.weight");
        errdefer position_embeddings.deinit();

        var token_type_embeddings = if (is_quantized)
            try loader.getTensorDequantized("roberta.embeddings.token_type_embeddings.weight")
        else
            try loader.getTensor("roberta.embeddings.token_type_embeddings.weight");
        errdefer token_type_embeddings.deinit();

        var embed_ln_gamma = if (is_quantized)
            try loader.getTensorDequantized("roberta.embeddings.LayerNorm.weight")
        else
            try loader.getTensor("roberta.embeddings.LayerNorm.weight");
        errdefer embed_ln_gamma.deinit();

        var embed_ln_beta = if (is_quantized)
            try loader.getTensorDequantized("roberta.embeddings.LayerNorm.bias")
        else
            try loader.getTensor("roberta.embeddings.LayerNorm.bias");
        errdefer embed_ln_beta.deinit();

        // Classification bias (always float32)
        var classifier_bias = if (is_quantized)
            try loader.getTensorDequantized("classifier.bias")
        else
            try loader.getTensor("classifier.bias");
        errdefer classifier_bias.deinit();

        // Load weights (format-dependent: float32, Q8_0, or Q4_0)
        var weights: WeightStorage = undefined;
        if (quant_format == .q8_0) {
            // Load Q8_0 quantized weights
            var blocks = try allocator.alloc(TransformerBlockWeightsQ8, config.num_layers);
            var loaded_blocks: usize = 0;
            errdefer {
                for (blocks[0..loaded_blocks]) |*b| b.deinit();
                allocator.free(blocks);
            }

            for (0..config.num_layers) |i| {
                blocks[i] = try loadTransformerBlockQ8(allocator, loader, i);
                loaded_blocks += 1;
            }

            // Classifier weight (quantized + transposed)
            var classifier_weight = try loadQuantizedTransposedQ8(allocator, loader, "classifier.weight");
            errdefer classifier_weight.deinit();

            weights = .{ .q8 = .{
                .blocks = blocks,
                .classifier_weight = classifier_weight,
            } };
        } else if (quant_format == .q4_0) {
            // Load Q4_0 quantized weights
            var blocks = try allocator.alloc(TransformerBlockWeightsQ4, config.num_layers);
            var loaded_blocks: usize = 0;
            errdefer {
                for (blocks[0..loaded_blocks]) |*b| b.deinit();
                allocator.free(blocks);
            }

            for (0..config.num_layers) |i| {
                blocks[i] = try loadTransformerBlockQ4(allocator, loader, i);
                loaded_blocks += 1;
            }

            // Classifier weight (quantized + transposed)
            var classifier_weight = try loadQuantizedTransposedQ4(allocator, loader, "classifier.weight");
            errdefer classifier_weight.deinit();

            weights = .{ .q4 = .{
                .blocks = blocks,
                .classifier_weight = classifier_weight,
            } };
        } else if (quant_format == .q8_k) {
            // Load Q8_K block-wise quantized weights
            var blocks = try allocator.alloc(TransformerBlockWeightsQ8K, config.num_layers);
            var loaded_blocks: usize = 0;
            errdefer {
                for (blocks[0..loaded_blocks]) |*b| b.deinit();
                allocator.free(blocks);
            }

            for (0..config.num_layers) |i| {
                blocks[i] = try loadTransformerBlockQ8K(allocator, loader, i);
                loaded_blocks += 1;
            }

            // Classifier weight (quantized + transposed)
            var classifier_weight = try loadQuantizedTransposedQ8K(allocator, loader, "classifier.weight");
            errdefer classifier_weight.deinit();

            weights = .{ .q8k = .{
                .blocks = blocks,
                .classifier_weight = classifier_weight,
            } };
        } else {
            // Load float32 weights
            var blocks = try allocator.alloc(TransformerBlockWeights, config.num_layers);
            var loaded_blocks: usize = 0;
            errdefer {
                for (blocks[0..loaded_blocks]) |*b| b.deinit();
                allocator.free(blocks);
            }

            for (0..config.num_layers) |i| {
                blocks[i] = try loadTransformerBlock(loader, i);
                loaded_blocks += 1;
            }

            // Classifier weight (transposed for matmul)
            var classifier_weight_pt = try loader.getTensor("classifier.weight");
            var classifier_weight = try ops.transpose(allocator, &classifier_weight_pt);
            classifier_weight_pt.deinit();
            errdefer classifier_weight.deinit();

            weights = .{ .f32 = .{
                .blocks = blocks,
                .classifier_weight = classifier_weight,
            } };
        }

        return Self{
            .allocator = allocator,
            .config = config,
            .quant_format = quant_format,
            .word_embeddings = word_embeddings,
            .position_embeddings = position_embeddings,
            .token_type_embeddings = token_type_embeddings,
            .embed_ln_gamma = embed_ln_gamma,
            .embed_ln_beta = embed_ln_beta,
            .weights = weights,
            .classifier_bias = classifier_bias,
        };
    }

    /// Auto-detect configuration from model weights
    fn detectConfig(loader: *ModelLoader) !XLMRobertaConfig {
        // Try to detect hidden_dim from word embeddings shape
        // Handle float32, Q8_0, Q4_0, and Q8_K formats
        const is_quantized = loader.quant_format == .q8_0 or loader.quant_format == .q4_0 or loader.quant_format == .q8_k;
        var word_emb = if (is_quantized)
            try loader.getTensorDequantized("roberta.embeddings.word_embeddings.weight")
        else
            try loader.getTensor("roberta.embeddings.word_embeddings.weight");
        defer word_emb.deinit();

        const hidden_dim = word_emb.shape[1];

        // Detect number of layers by checking which layers exist
        var num_layers: usize = 0;
        var buf: [128]u8 = undefined;
        while (num_layers < 48) : (num_layers += 1) {
            const layer_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.query.weight", .{num_layers}) catch unreachable;
            if (is_quantized) {
                var tensor = loader.getTensorDequantized(layer_name) catch break;
                tensor.deinit();
            } else {
                var tensor = loader.getTensor(layer_name) catch break;
                tensor.deinit();
            }
        }

        // Determine config based on detected values
        if (hidden_dim == 1024 and num_layers == 24) {
            return CONFIG_LARGE;
        } else {
            return XLMRobertaConfig{
                .hidden_dim = hidden_dim,
                .intermediate_dim = hidden_dim * 4,
                .num_layers = num_layers,
                .num_heads = if (hidden_dim == 1024) 16 else 12,
            };
        }
    }

    /// Load a single transformer block from the model file
    /// XLM-RoBERTa uses "roberta.encoder.layer.X.*" naming
    /// NOTE: Weights are PRE-TRANSPOSED at load time for optimal SIMD matmul performance.
    fn loadTransformerBlock(loader: *ModelLoader, layer_idx: usize) !TransformerBlockWeights {
        var buf: [128]u8 = undefined;

        // Attention weights (query, key, value) - pre-transpose for SIMD matmul
        const q_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.query.weight", .{layer_idx}) catch unreachable;
        var q_weight_pt = try loader.getTensor(q_weight_name);
        var q_weight = try ops.transpose(loader.allocator, &q_weight_pt);
        q_weight_pt.deinit();
        errdefer q_weight.deinit();

        const k_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.key.weight", .{layer_idx}) catch unreachable;
        var k_weight_pt = try loader.getTensor(k_weight_name);
        var k_weight = try ops.transpose(loader.allocator, &k_weight_pt);
        k_weight_pt.deinit();
        errdefer k_weight.deinit();

        const v_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.value.weight", .{layer_idx}) catch unreachable;
        var v_weight_pt = try loader.getTensor(v_weight_name);
        var v_weight = try ops.transpose(loader.allocator, &v_weight_pt);
        v_weight_pt.deinit();
        errdefer v_weight.deinit();

        // Output projection - pre-transpose for SIMD matmul
        const o_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.dense.weight", .{layer_idx}) catch unreachable;
        var o_weight_pt = try loader.getTensor(o_weight_name);
        var o_weight = try ops.transpose(loader.allocator, &o_weight_pt);
        o_weight_pt.deinit();
        errdefer o_weight.deinit();

        // Biases
        const q_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.query.bias", .{layer_idx}) catch unreachable;
        var q_bias = try loader.getTensor(q_bias_name);
        errdefer q_bias.deinit();

        const k_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.key.bias", .{layer_idx}) catch unreachable;
        var k_bias = try loader.getTensor(k_bias_name);
        errdefer k_bias.deinit();

        const v_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.value.bias", .{layer_idx}) catch unreachable;
        var v_bias = try loader.getTensor(v_bias_name);
        errdefer v_bias.deinit();

        const o_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.dense.bias", .{layer_idx}) catch unreachable;
        var o_bias = try loader.getTensor(o_bias_name);
        errdefer o_bias.deinit();

        // Attention layer norm
        const attn_ln_gamma_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.LayerNorm.weight", .{layer_idx}) catch unreachable;
        var attn_ln_gamma = try loader.getTensor(attn_ln_gamma_name);
        errdefer attn_ln_gamma.deinit();

        const attn_ln_beta_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.LayerNorm.bias", .{layer_idx}) catch unreachable;
        var attn_ln_beta = try loader.getTensor(attn_ln_beta_name);
        errdefer attn_ln_beta.deinit();

        // FFN weights (intermediate and output) - pre-transpose for SIMD matmul
        const ff1_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.intermediate.dense.weight", .{layer_idx}) catch unreachable;
        var ff1_weight_pt = try loader.getTensor(ff1_weight_name);
        var ff1_weight = try ops.transpose(loader.allocator, &ff1_weight_pt);
        ff1_weight_pt.deinit();
        errdefer ff1_weight.deinit();

        const ff1_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.intermediate.dense.bias", .{layer_idx}) catch unreachable;
        var ff1_bias = try loader.getTensor(ff1_bias_name);
        errdefer ff1_bias.deinit();

        const ff2_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.dense.weight", .{layer_idx}) catch unreachable;
        var ff2_weight_pt = try loader.getTensor(ff2_weight_name);
        var ff2_weight = try ops.transpose(loader.allocator, &ff2_weight_pt);
        ff2_weight_pt.deinit();
        errdefer ff2_weight.deinit();

        const ff2_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.dense.bias", .{layer_idx}) catch unreachable;
        var ff2_bias = try loader.getTensor(ff2_bias_name);
        errdefer ff2_bias.deinit();

        // Output layer norm
        const ff_ln_gamma_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.LayerNorm.weight", .{layer_idx}) catch unreachable;
        var ff_ln_gamma = try loader.getTensor(ff_ln_gamma_name);
        errdefer ff_ln_gamma.deinit();

        const ff_ln_beta_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.LayerNorm.bias", .{layer_idx}) catch unreachable;
        var ff_ln_beta = try loader.getTensor(ff_ln_beta_name);
        errdefer ff_ln_beta.deinit();

        return TransformerBlockWeights{
            .attention = AttentionWeights{
                .q_weight = q_weight,
                .k_weight = k_weight,
                .v_weight = v_weight,
                .o_weight = o_weight,
                .q_bias = q_bias,
                .k_bias = k_bias,
                .v_bias = v_bias,
                .o_bias = o_bias,
            },
            .attn_ln_gamma = attn_ln_gamma,
            .attn_ln_beta = attn_ln_beta,
            .ff_linear1_weight = ff1_weight,
            .ff_linear1_bias = ff1_bias,
            .ff_linear2_weight = ff2_weight,
            .ff_linear2_bias = ff2_bias,
            .ff_ln_gamma = ff_ln_gamma,
            .ff_ln_beta = ff_ln_beta,
        };
    }

    /// Load a single Q8_0 quantized transformer block
    fn loadTransformerBlockQ8(allocator: std.mem.Allocator, loader: *ModelLoader, layer_idx: usize) !TransformerBlockWeightsQ8 {
        var buf: [128]u8 = undefined;

        // Attention weights (transposed for SIMD matmul)
        const q_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.query.weight", .{layer_idx}) catch unreachable;
        var q_weight = try loadQuantizedTransposedQ8(allocator, loader, q_weight_name);
        errdefer q_weight.deinit();

        const k_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.key.weight", .{layer_idx}) catch unreachable;
        var k_weight = try loadQuantizedTransposedQ8(allocator, loader, k_weight_name);
        errdefer k_weight.deinit();

        const v_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.value.weight", .{layer_idx}) catch unreachable;
        var v_weight = try loadQuantizedTransposedQ8(allocator, loader, v_weight_name);
        errdefer v_weight.deinit();

        const o_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.dense.weight", .{layer_idx}) catch unreachable;
        var o_weight = try loadQuantizedTransposedQ8(allocator, loader, o_weight_name);
        errdefer o_weight.deinit();

        // Biases (dequantized to float32)
        const q_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.query.bias", .{layer_idx}) catch unreachable;
        var q_bias = try loader.getTensorDequantized(q_bias_name);
        errdefer q_bias.deinit();

        const k_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.key.bias", .{layer_idx}) catch unreachable;
        var k_bias = try loader.getTensorDequantized(k_bias_name);
        errdefer k_bias.deinit();

        const v_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.value.bias", .{layer_idx}) catch unreachable;
        var v_bias = try loader.getTensorDequantized(v_bias_name);
        errdefer v_bias.deinit();

        const o_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.dense.bias", .{layer_idx}) catch unreachable;
        var o_bias = try loader.getTensorDequantized(o_bias_name);
        errdefer o_bias.deinit();

        // Layer norm (float32)
        const attn_ln_gamma_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.LayerNorm.weight", .{layer_idx}) catch unreachable;
        var attn_ln_gamma = try loader.getTensorDequantized(attn_ln_gamma_name);
        errdefer attn_ln_gamma.deinit();

        const attn_ln_beta_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.LayerNorm.bias", .{layer_idx}) catch unreachable;
        var attn_ln_beta = try loader.getTensorDequantized(attn_ln_beta_name);
        errdefer attn_ln_beta.deinit();

        // FFN weights (transposed and quantized)
        const ff1_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.intermediate.dense.weight", .{layer_idx}) catch unreachable;
        var ff1_weight = try loadQuantizedTransposedQ8(allocator, loader, ff1_weight_name);
        errdefer ff1_weight.deinit();

        const ff1_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.intermediate.dense.bias", .{layer_idx}) catch unreachable;
        var ff1_bias = try loader.getTensorDequantized(ff1_bias_name);
        errdefer ff1_bias.deinit();

        const ff2_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.dense.weight", .{layer_idx}) catch unreachable;
        var ff2_weight = try loadQuantizedTransposedQ8(allocator, loader, ff2_weight_name);
        errdefer ff2_weight.deinit();

        const ff2_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.dense.bias", .{layer_idx}) catch unreachable;
        var ff2_bias = try loader.getTensorDequantized(ff2_bias_name);
        errdefer ff2_bias.deinit();

        // Output layer norm (float32)
        const ff_ln_gamma_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.LayerNorm.weight", .{layer_idx}) catch unreachable;
        var ff_ln_gamma = try loader.getTensorDequantized(ff_ln_gamma_name);
        errdefer ff_ln_gamma.deinit();

        const ff_ln_beta_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.LayerNorm.bias", .{layer_idx}) catch unreachable;
        var ff_ln_beta = try loader.getTensorDequantized(ff_ln_beta_name);
        errdefer ff_ln_beta.deinit();

        return TransformerBlockWeightsQ8{
            .attention = AttentionWeightsQ8{
                .q_weight = q_weight,
                .k_weight = k_weight,
                .v_weight = v_weight,
                .o_weight = o_weight,
                .q_bias = q_bias,
                .k_bias = k_bias,
                .v_bias = v_bias,
                .o_bias = o_bias,
            },
            .attn_ln_gamma = attn_ln_gamma,
            .attn_ln_beta = attn_ln_beta,
            .ff_linear1_weight = ff1_weight,
            .ff_linear1_bias = ff1_bias,
            .ff_linear2_weight = ff2_weight,
            .ff_linear2_bias = ff2_bias,
            .ff_ln_gamma = ff_ln_gamma,
            .ff_ln_beta = ff_ln_beta,
        };
    }

    /// Load a single Q4_0 quantized transformer block
    fn loadTransformerBlockQ4(allocator: std.mem.Allocator, loader: *ModelLoader, layer_idx: usize) !TransformerBlockWeightsQ4 {
        var buf: [128]u8 = undefined;

        // Attention weights (transposed for SIMD matmul)
        const q_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.query.weight", .{layer_idx}) catch unreachable;
        var q_weight = try loadQuantizedTransposedQ4(allocator, loader, q_weight_name);
        errdefer q_weight.deinit();

        const k_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.key.weight", .{layer_idx}) catch unreachable;
        var k_weight = try loadQuantizedTransposedQ4(allocator, loader, k_weight_name);
        errdefer k_weight.deinit();

        const v_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.value.weight", .{layer_idx}) catch unreachable;
        var v_weight = try loadQuantizedTransposedQ4(allocator, loader, v_weight_name);
        errdefer v_weight.deinit();

        const o_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.dense.weight", .{layer_idx}) catch unreachable;
        var o_weight = try loadQuantizedTransposedQ4(allocator, loader, o_weight_name);
        errdefer o_weight.deinit();

        // Biases (dequantized to float32)
        const q_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.query.bias", .{layer_idx}) catch unreachable;
        var q_bias = try loader.getTensorDequantized(q_bias_name);
        errdefer q_bias.deinit();

        const k_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.key.bias", .{layer_idx}) catch unreachable;
        var k_bias = try loader.getTensorDequantized(k_bias_name);
        errdefer k_bias.deinit();

        const v_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.value.bias", .{layer_idx}) catch unreachable;
        var v_bias = try loader.getTensorDequantized(v_bias_name);
        errdefer v_bias.deinit();

        const o_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.dense.bias", .{layer_idx}) catch unreachable;
        var o_bias = try loader.getTensorDequantized(o_bias_name);
        errdefer o_bias.deinit();

        // Attention layer norm (dequantized to float32)
        const attn_ln_gamma_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.LayerNorm.weight", .{layer_idx}) catch unreachable;
        var attn_ln_gamma = try loader.getTensorDequantized(attn_ln_gamma_name);
        errdefer attn_ln_gamma.deinit();

        const attn_ln_beta_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.LayerNorm.bias", .{layer_idx}) catch unreachable;
        var attn_ln_beta = try loader.getTensorDequantized(attn_ln_beta_name);
        errdefer attn_ln_beta.deinit();

        // FFN weights (transposed for SIMD matmul)
        const ff1_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.intermediate.dense.weight", .{layer_idx}) catch unreachable;
        var ff1_weight = try loadQuantizedTransposedQ4(allocator, loader, ff1_weight_name);
        errdefer ff1_weight.deinit();

        const ff1_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.intermediate.dense.bias", .{layer_idx}) catch unreachable;
        var ff1_bias = try loader.getTensorDequantized(ff1_bias_name);
        errdefer ff1_bias.deinit();

        const ff2_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.dense.weight", .{layer_idx}) catch unreachable;
        var ff2_weight = try loadQuantizedTransposedQ4(allocator, loader, ff2_weight_name);
        errdefer ff2_weight.deinit();

        const ff2_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.dense.bias", .{layer_idx}) catch unreachable;
        var ff2_bias = try loader.getTensorDequantized(ff2_bias_name);
        errdefer ff2_bias.deinit();

        // Output layer norm (dequantized to float32)
        const ff_ln_gamma_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.LayerNorm.weight", .{layer_idx}) catch unreachable;
        var ff_ln_gamma = try loader.getTensorDequantized(ff_ln_gamma_name);
        errdefer ff_ln_gamma.deinit();

        const ff_ln_beta_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.LayerNorm.bias", .{layer_idx}) catch unreachable;
        var ff_ln_beta = try loader.getTensorDequantized(ff_ln_beta_name);
        errdefer ff_ln_beta.deinit();

        return TransformerBlockWeightsQ4{
            .attention = AttentionWeightsQ4{
                .q_weight = q_weight,
                .k_weight = k_weight,
                .v_weight = v_weight,
                .o_weight = o_weight,
                .q_bias = q_bias,
                .k_bias = k_bias,
                .v_bias = v_bias,
                .o_bias = o_bias,
            },
            .attn_ln_gamma = attn_ln_gamma,
            .attn_ln_beta = attn_ln_beta,
            .ff_linear1_weight = ff1_weight,
            .ff_linear1_bias = ff1_bias,
            .ff_linear2_weight = ff2_weight,
            .ff_linear2_bias = ff2_bias,
            .ff_ln_gamma = ff_ln_gamma,
            .ff_ln_beta = ff_ln_beta,
        };
    }

    /// Load a single Q8_K block-wise quantized transformer block
    fn loadTransformerBlockQ8K(allocator: std.mem.Allocator, loader: *ModelLoader, layer_idx: usize) !TransformerBlockWeightsQ8K {
        var buf: [128]u8 = undefined;

        // Attention weights (transposed for SIMD matmul)
        const q_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.query.weight", .{layer_idx}) catch unreachable;
        var q_weight = try loadQuantizedTransposedQ8K(allocator, loader, q_weight_name);
        errdefer q_weight.deinit();

        const k_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.key.weight", .{layer_idx}) catch unreachable;
        var k_weight = try loadQuantizedTransposedQ8K(allocator, loader, k_weight_name);
        errdefer k_weight.deinit();

        const v_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.value.weight", .{layer_idx}) catch unreachable;
        var v_weight = try loadQuantizedTransposedQ8K(allocator, loader, v_weight_name);
        errdefer v_weight.deinit();

        const o_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.dense.weight", .{layer_idx}) catch unreachable;
        var o_weight = try loadQuantizedTransposedQ8K(allocator, loader, o_weight_name);
        errdefer o_weight.deinit();

        // Biases (dequantized to float32)
        const q_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.query.bias", .{layer_idx}) catch unreachable;
        var q_bias = try loader.getTensorDequantized(q_bias_name);
        errdefer q_bias.deinit();

        const k_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.key.bias", .{layer_idx}) catch unreachable;
        var k_bias = try loader.getTensorDequantized(k_bias_name);
        errdefer k_bias.deinit();

        const v_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.self.value.bias", .{layer_idx}) catch unreachable;
        var v_bias = try loader.getTensorDequantized(v_bias_name);
        errdefer v_bias.deinit();

        const o_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.dense.bias", .{layer_idx}) catch unreachable;
        var o_bias = try loader.getTensorDequantized(o_bias_name);
        errdefer o_bias.deinit();

        // Layer norm (float32)
        const attn_ln_gamma_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.LayerNorm.weight", .{layer_idx}) catch unreachable;
        var attn_ln_gamma = try loader.getTensorDequantized(attn_ln_gamma_name);
        errdefer attn_ln_gamma.deinit();

        const attn_ln_beta_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.attention.output.LayerNorm.bias", .{layer_idx}) catch unreachable;
        var attn_ln_beta = try loader.getTensorDequantized(attn_ln_beta_name);
        errdefer attn_ln_beta.deinit();

        // FFN weights (transposed and quantized)
        const ff1_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.intermediate.dense.weight", .{layer_idx}) catch unreachable;
        var ff1_weight = try loadQuantizedTransposedQ8K(allocator, loader, ff1_weight_name);
        errdefer ff1_weight.deinit();

        const ff1_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.intermediate.dense.bias", .{layer_idx}) catch unreachable;
        var ff1_bias = try loader.getTensorDequantized(ff1_bias_name);
        errdefer ff1_bias.deinit();

        const ff2_weight_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.dense.weight", .{layer_idx}) catch unreachable;
        var ff2_weight = try loadQuantizedTransposedQ8K(allocator, loader, ff2_weight_name);
        errdefer ff2_weight.deinit();

        const ff2_bias_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.dense.bias", .{layer_idx}) catch unreachable;
        var ff2_bias = try loader.getTensorDequantized(ff2_bias_name);
        errdefer ff2_bias.deinit();

        // Output layer norm (float32)
        const ff_ln_gamma_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.LayerNorm.weight", .{layer_idx}) catch unreachable;
        var ff_ln_gamma = try loader.getTensorDequantized(ff_ln_gamma_name);
        errdefer ff_ln_gamma.deinit();

        const ff_ln_beta_name = std.fmt.bufPrint(&buf, "roberta.encoder.layer.{d}.output.LayerNorm.bias", .{layer_idx}) catch unreachable;
        var ff_ln_beta = try loader.getTensorDequantized(ff_ln_beta_name);
        errdefer ff_ln_beta.deinit();

        return TransformerBlockWeightsQ8K{
            .attention = AttentionWeightsQ8K{
                .q_weight = q_weight,
                .k_weight = k_weight,
                .v_weight = v_weight,
                .o_weight = o_weight,
                .q_bias = q_bias,
                .k_bias = k_bias,
                .v_bias = v_bias,
                .o_bias = o_bias,
            },
            .attn_ln_gamma = attn_ln_gamma,
            .attn_ln_beta = attn_ln_beta,
            .ff_linear1_weight = ff1_weight,
            .ff_linear1_bias = ff1_bias,
            .ff_linear2_weight = ff2_weight,
            .ff_linear2_bias = ff2_bias,
            .ff_ln_gamma = ff_ln_gamma,
            .ff_ln_beta = ff_ln_beta,
        };
    }

    /// Run inference on token IDs
    /// Returns predicted punctuation labels for each token
    pub fn forward(self: *Self, input_ids: []const u32) ![]PunctuationLabel {
        const seq_len = input_ids.len;
        const config = self.config;
        const transformer_config = config.getTransformerConfig();


        // 1. Embeddings: word + position + token_type
        var word_emb = try ops.embedding(self.allocator, input_ids, &self.word_embeddings);
        defer word_emb.deinit();

        // Position IDs: RoBERTa uses offset of 2 (positions 0-1 reserved for padding)
        // For sequence [0, 33600, 31, 8999, 2], position IDs are [2, 3, 4, 5, 6]
        const position_ids = try self.allocator.alloc(u32, seq_len);
        defer self.allocator.free(position_ids);
        const POSITION_OFFSET: u32 = 2; // RoBERTa-specific offset
        for (position_ids, 0..) |*p, i| {
            p.* = @as(u32, @intCast(i)) + POSITION_OFFSET;
        }

        var pos_emb = try ops.embedding(self.allocator, position_ids, &self.position_embeddings);
        defer pos_emb.deinit();

        // Token type IDs: all zeros for single sequence
        const token_type_ids = try self.allocator.alloc(u32, seq_len);
        defer self.allocator.free(token_type_ids);
        @memset(token_type_ids, 0);

        var type_emb = try ops.embedding(self.allocator, token_type_ids, &self.token_type_embeddings);
        defer type_emb.deinit();

        // Combine embeddings
        try ops.addInPlace(&word_emb, &pos_emb);
        try ops.addInPlace(&word_emb, &type_emb);

        // Embedding layer norm
        var hidden = try ops.layerNorm(
            self.allocator,
            &word_emb,
            &self.embed_ln_gamma,
            &self.embed_ln_beta,
            config.layer_norm_eps,
        );
        defer hidden.deinit();

        // 2. Run through transformer blocks (branch on weight format)
        // 3. Classification head
        var logits: Tensor = undefined;
        switch (self.weights) {
            .f32 => |f| {
                // Float32 path
                for (f.blocks) |*block| {
                    const new_hidden = try transformer.transformerBlockF32(
                        self.allocator,
                        &hidden,
                        block,
                        transformer_config,
                    );
                    hidden.deinit();
                    hidden = new_hidden;
                }
                logits = try ops.matmul(self.allocator, &hidden, &f.classifier_weight);
            },
            .q8 => |q| {
                // Quantized Q8_0 path
                for (q.blocks) |*block| {
                    const new_hidden = try transformer.transformerBlockQ8(
                        self.allocator,
                        &hidden,
                        block,
                        transformer_config,
                    );
                    hidden.deinit();
                    hidden = new_hidden;
                }
                logits = try quant.matmulF32Q8Simd(self.allocator, &hidden, &q.classifier_weight);
            },
            .q4 => |q| {
                // Quantized Q4_0 path
                for (q.blocks) |*block| {
                    const new_hidden = try transformer.transformerBlockQ4(
                        self.allocator,
                        &hidden,
                        block,
                        transformer_config,
                    );
                    hidden.deinit();
                    hidden = new_hidden;
                }
                logits = try quant.matmulF32Q4Simd(self.allocator, &hidden, &q.classifier_weight);
            },
            .q8k => |q| {
                // Quantized Q8_K block-wise path
                for (q.blocks) |*block| {
                    const new_hidden = try transformer.transformerBlockQ8K(
                        self.allocator,
                        &hidden,
                        block,
                        transformer_config,
                    );
                    hidden.deinit();
                    hidden = new_hidden;
                }
                logits = try quant.matmulF32Q8KSimd(self.allocator, &hidden, &q.classifier_weight);
            },
        }
        defer logits.deinit();
        try ops.addBiasInPlace(&logits, &self.classifier_bias);

        // 4. Argmax per token
        var labels = try self.allocator.alloc(PunctuationLabel, seq_len);
        errdefer self.allocator.free(labels);

        const num_labels = config.num_labels;
        for (0..seq_len) |i| {
            const row_start = i * num_labels;
            var max_idx: usize = 0;
            var max_val = logits.data[row_start];
            for (1..num_labels) |j| {
                if (logits.data[row_start + j] > max_val) {
                    max_val = logits.data[row_start + j];
                    max_idx = j;
                }
            }
            labels[i] = @enumFromInt(@as(u8, @intCast(max_idx)));
        }

        return labels;
    }

    /// Apply predicted labels to reconstruct punctuated text
    pub fn applyLabels(_: *Self, tokens: []const []const u8, labels: []const PunctuationLabel, allocator: std.mem.Allocator) ![]u8 {
        var result: std.ArrayList(u8) = .{};
        errdefer result.deinit(allocator);

        var capitalize_next = true; // Start with capital
        var pending_punct: ?u8 = null; // Accumulate punctuation for word boundaries

        for (tokens, labels, 0..) |token, label, idx| {
            // Skip special tokens
            if (token.len >= 3 and token[0] == '<' and token[token.len - 1] == '>') {
                continue;
            }

            // Check if this token starts a new word (has "▁" prefix)
            const is_word_start = token.len >= 3 and std.mem.startsWith(u8, token, "▁");

            // Handle SentencePiece "▁" prefix (word boundary)
            var word = token;
            var add_space = false;
            if (is_word_start) {
                // Before starting new word, add any pending punctuation from previous word
                if (pending_punct) |punct| {
                    try result.append(allocator, punct);
                    pending_punct = null;
                }

                word = token[3..]; // Skip the 3-byte UTF-8 "▁"
                add_space = result.items.len > 0;
            }

            if (add_space) {
                try result.append(allocator, ' ');
            }

            // Add word (with optional capitalization)
            for (word, 0..) |c, i| {
                if (i == 0 and capitalize_next and c >= 'a' and c <= 'z') {
                    try result.append(allocator, c - 32); // Uppercase
                    capitalize_next = false;
                } else {
                    try result.append(allocator, c);
                }
            }

            // Only apply punctuation at word boundaries or end of sequence
            const is_last_token = idx == tokens.len - 1;
            const next_is_word_start = if (idx + 1 < tokens.len)
                (tokens[idx + 1].len >= 3 and std.mem.startsWith(u8, tokens[idx + 1], "▁"))
            else false;

            if (is_word_start or is_last_token or next_is_word_start) {
                // This completes a word, so we can add punctuation
                if (label.toChar()) |punct| {
                    pending_punct = punct;
                    if (is_last_token) {
                        try result.append(allocator, punct);
                        pending_punct = null;
                    }
                }
                if (label.shouldCapitalizeNext()) {
                    capitalize_next = true;
                }
            }
        }

        // Add any remaining punctuation
        if (pending_punct) |punct| {
            try result.append(allocator, punct);
        }

        return result.toOwnedSlice(allocator);
    }

    pub fn deinit(self: *Self) void {
        self.word_embeddings.deinit();
        self.position_embeddings.deinit();
        self.token_type_embeddings.deinit();
        self.embed_ln_gamma.deinit();
        self.embed_ln_beta.deinit();
        self.weights.deinit(self.allocator);
        self.classifier_bias.deinit();
    }
};

/// Helper: Load a quantized tensor and transpose it for matmul
/// PyTorch stores [out, in], we need [in, out] for A @ W
/// This is done at load time (once), so overhead is acceptable.
pub fn loadQuantizedTransposedQ8(allocator: std.mem.Allocator, loader: *ModelLoader, name: []const u8) !QuantizedTensorQ8 {
    // Load as Q8_0
    var qt = try loader.getQuantizedTensorQ8(name);
    errdefer qt.deinit();

    // For transpose: dequantize, transpose, re-quantize
    // Dequantize to float32
    var tensor_f32 = try Tensor.init(allocator, qt.shape);
    defer tensor_f32.deinit();

    for (tensor_f32.data, qt.data) |*out, q| {
        out.* = @as(f32, @floatFromInt(q)) * qt.scale;
    }
    qt.deinit(); // Free original after copying

    // Transpose
    var transposed = try ops.transpose(allocator, &tensor_f32);
    defer transposed.deinit();

    // Re-quantize
    return quant.quantizeQ8(allocator, &transposed);
}

/// Helper: Load a Q4_0 quantized tensor and transpose it for matmul
/// PyTorch stores [out, in], we need [in, out] for A @ W
/// This is done at load time (once), so overhead is acceptable.
pub fn loadQuantizedTransposedQ4(allocator: std.mem.Allocator, loader: *ModelLoader, name: []const u8) !QuantizedTensorQ4 {
    // Load as Q4_0
    var qt = try loader.getQuantizedTensorQ4(name);
    errdefer qt.deinit();

    // For transpose: dequantize, transpose, re-quantize
    // Dequantize to float32
    var tensor_f32 = try Tensor.init(allocator, qt.shape);
    defer tensor_f32.deinit();

    // Q4 is packed: 2 values per byte
    const total_elements = qt.shape[0] * qt.shape[1];
    for (0..total_elements) |i| {
        const byte_idx = i / 2;
        const packed_byte = qt.data[byte_idx];
        const nibble: u4 = if (i % 2 == 0)
            @truncate(packed_byte & 0x0F)
        else
            @truncate((packed_byte >> 4) & 0x0F);
        // Sign-extend from 4-bit
        const value: i8 = if (nibble & 0x08 != 0)
            @as(i8, @intCast(nibble)) - 16
        else
            @as(i8, @intCast(nibble));
        tensor_f32.data[i] = @as(f32, @floatFromInt(value)) * qt.scale;
    }
    qt.deinit(); // Free original after copying

    // Transpose
    var transposed = try ops.transpose(allocator, &tensor_f32);
    defer transposed.deinit();

    // Re-quantize to Q4
    return quant.quantizeQ4(allocator, &transposed);
}

/// Helper: Load a Q8_K block-wise quantized tensor and transpose it for matmul
/// PyTorch stores [out, in], we need [in, out] for A @ W
/// This is done at load time (once), so overhead is acceptable.
pub fn loadQuantizedTransposedQ8K(allocator: std.mem.Allocator, loader: *ModelLoader, name: []const u8) !QuantizedTensorQ8K {
    // Load as Q8_K
    var qt = try loader.getQuantizedTensorQ8K(name);
    errdefer qt.deinit();

    // For transpose: dequantize, transpose, re-quantize
    // Dequantize to float32
    var tensor_f32 = try Tensor.init(allocator, qt.shape);
    defer tensor_f32.deinit();

    // Q8_K has per-block scales
    const total_elements = qt.shape[0] * qt.shape[1];
    for (0..total_elements) |i| {
        const block_idx = i / qt.block_size;
        const scale = qt.scales[block_idx];
        tensor_f32.data[i] = @as(f32, @floatFromInt(qt.data[i])) * scale;
    }
    qt.deinit(); // Free original after copying

    // Transpose
    var transposed = try ops.transpose(allocator, &tensor_f32);
    defer transposed.deinit();

    // Re-quantize to Q8_K
    return quant.quantizeQ8K(allocator, &transposed);
}

// Tests
test "XLMRobertaConfig presets" {
    const base = CONFIG_BASE;
    try std.testing.expectEqual(@as(usize, 768), base.hidden_dim);
    try std.testing.expectEqual(@as(usize, 12), base.num_layers);

    const large = CONFIG_LARGE;
    try std.testing.expectEqual(@as(usize, 1024), large.hidden_dim);
    try std.testing.expectEqual(@as(usize, 24), large.num_layers);
}

test "PunctuationLabel conversion" {
    try std.testing.expectEqual(@as(?u8, null), PunctuationLabel.O.toChar());
    try std.testing.expectEqual(@as(?u8, ','), PunctuationLabel.COMMA.toChar());
    try std.testing.expectEqual(@as(?u8, '.'), PunctuationLabel.PERIOD.toChar());
    try std.testing.expectEqual(@as(?u8, '?'), PunctuationLabel.QUESTION.toChar());

    try std.testing.expect(!PunctuationLabel.O.shouldCapitalizeNext());
    try std.testing.expect(!PunctuationLabel.COMMA.shouldCapitalizeNext());
    try std.testing.expect(PunctuationLabel.PERIOD_U.shouldCapitalizeNext());
    try std.testing.expect(PunctuationLabel.QUESTION.shouldCapitalizeNext());
}

test "special tokens" {
    try std.testing.expectEqual(@as(u32, 0), SpecialTokens.BOS);
    try std.testing.expectEqual(@as(u32, 1), SpecialTokens.PAD);
    try std.testing.expectEqual(@as(u32, 2), SpecialTokens.EOS);
    try std.testing.expectEqual(@as(u32, 3), SpecialTokens.UNK);
}

test "unescape token" {
    const allocator = std.testing.allocator;

    const result1 = try unescapeToken(allocator, "hello\\nworld");
    defer allocator.free(result1);
    try std.testing.expectEqualStrings("hello\nworld", result1);

    const result2 = try unescapeToken(allocator, "tab\\there");
    defer allocator.free(result2);
    try std.testing.expectEqualStrings("tab\there", result2);

    const result3 = try unescapeToken(allocator, "back\\\\slash");
    defer allocator.free(result3);
    try std.testing.expectEqualStrings("back\\slash", result3);
}

test "UTF-8 boundaries" {
    const text = "hello▁world"; // Contains 3-byte UTF-8 character
    try std.testing.expectEqual(@as(usize, 5), prevCharBoundary(text, 8)); // Before ▁
    try std.testing.expectEqual(@as(usize, 8), nextCharBoundary(text, 5)); // After ▁
}

// =============================================================================
// Punctuation Model Wrapper (combines XLM-RoBERTa + Tokenizer)
// =============================================================================

/// Combined Punctuation Model (XLM-RoBERTa + Tokenizer)
pub const PunctuationModel = struct {
    allocator: std.mem.Allocator,
    model: XLMRobertaModel,
    tokenizer: Tokenizer,

    const Self = @This();

    /// Initialize from file paths
    pub fn init(allocator: std.mem.Allocator, weights_path: []const u8, vocab_path: []const u8) !Self {
        var model = try XLMRobertaModel.init(allocator, weights_path);
        errdefer model.deinit();

        var tokenizer = try Tokenizer.init(allocator, vocab_path);
        errdefer tokenizer.deinit();

        return Self{
            .allocator = allocator,
            .model = model,
            .tokenizer = tokenizer,
        };
    }

    /// Initialize from embedded bytes (for WASM bundled builds)
    pub fn initFromBytes(allocator: std.mem.Allocator, weights_bytes: []const u8, vocab_bytes: []const u8) !Self {
        var model = try XLMRobertaModel.initFromBytes(allocator, weights_bytes);
        errdefer model.deinit();

        var tokenizer = try Tokenizer.initFromString(allocator, vocab_bytes);
        errdefer tokenizer.deinit();

        return Self{
            .allocator = allocator,
            .model = model,
            .tokenizer = tokenizer,
        };
    }

    /// Process input text and return punctuated text
    pub fn process(self: *Self, input_text: []const u8) ![]u8 {
        // Tokenize
        const encoded = try self.tokenizer.encode(input_text);
        defer self.allocator.free(encoded.ids);
        defer {
            for (encoded.tokens) |token| {
                self.allocator.free(token);
            }
            self.allocator.free(encoded.tokens);
        }

        // Run model
        const labels = try self.model.forward(encoded.ids);
        defer self.allocator.free(labels);

        // Apply labels
        return try self.model.applyLabels(encoded.tokens, labels, self.allocator);
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
        self.tokenizer.deinit();
    }
};
