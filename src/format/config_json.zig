// src/format/config_json.zig
//
// HuggingFace `config.json` reader.
//
// One generic loader (returns std.json.Value backed by an arena), plus
// small typed views per architecture family. Architecture-specific fields
// live next to the architecture that consumes them.

const std = @import("std");

pub const Error = error{
    FileTooLarge,
    InvalidJson,
    MissingField,
    WrongType,
} || std.fs.File.OpenError || std.mem.Allocator.Error || std.fs.File.ReadError;

const MAX_CONFIG_BYTES: usize = 1 * 1024 * 1024; // 1 MiB ceiling — real configs are <10 KiB

/// Parsed HF config.json kept alive by an arena.
pub const ConfigJson = struct {
    arena: std.heap.ArenaAllocator,
    root: std.json.Value,

    const Self = @This();

    pub fn initFromFile(allocator: std.mem.Allocator, path: []const u8) !Self {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const size = try file.getEndPos();
        if (size > MAX_CONFIG_BYTES) return Error.FileTooLarge;

        const buf = try allocator.alloc(u8, size);
        defer allocator.free(buf);

        const n = try file.readAll(buf);
        if (n != size) return Error.InvalidJson;

        return initFromSlice(allocator, buf);
    }

    pub fn initFromSlice(allocator: std.mem.Allocator, slice: []const u8) !Self {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const root = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), slice, .{}) catch
            return Error.InvalidJson;
        if (root != .object) return Error.InvalidJson;

        return Self{ .arena = arena, .root = root };
    }

    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    pub fn getString(self: *const Self, key: []const u8) ?[]const u8 {
        const v = self.root.object.get(key) orelse return null;
        return if (v == .string) v.string else null;
    }

    pub fn getInt(self: *const Self, key: []const u8) ?i64 {
        const v = self.root.object.get(key) orelse return null;
        return if (v == .integer) v.integer else null;
    }

    pub fn getFloat(self: *const Self, key: []const u8) ?f64 {
        const v = self.root.object.get(key) orelse return null;
        return switch (v) {
            .float => v.float,
            .integer => @floatFromInt(v.integer),
            else => null,
        };
    }

    pub fn getBool(self: *const Self, key: []const u8) ?bool {
        const v = self.root.object.get(key) orelse return null;
        return if (v == .bool) v.bool else null;
    }
};

/// Typed view of the Llama-family config.json fields.
/// Matches HuggingFace's LlamaConfig / Llama-3 / N-ATLaS / InkubaLM / AfroLlama.
pub const LlamaConfig = struct {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    /// Optional — falls back to hidden_size / num_attention_heads.
    head_dim: usize,
    /// Whether lm_head ties to the embedding matrix (HF default true for Llama).
    tie_word_embeddings: bool,

    pub fn fromConfig(cfg: *const ConfigJson) !LlamaConfig {
        const hidden_size = cfg.getInt("hidden_size") orelse return Error.MissingField;
        const num_layers = cfg.getInt("num_hidden_layers") orelse return Error.MissingField;
        const num_heads = cfg.getInt("num_attention_heads") orelse return Error.MissingField;
        const num_kv_heads = cfg.getInt("num_key_value_heads") orelse num_heads;
        const intermediate = cfg.getInt("intermediate_size") orelse return Error.MissingField;
        const vocab = cfg.getInt("vocab_size") orelse return Error.MissingField;
        const max_pos = cfg.getInt("max_position_embeddings") orelse 2048;
        const head_dim_field = cfg.getInt("head_dim");

        return LlamaConfig{
            .hidden_size = @intCast(hidden_size),
            .num_hidden_layers = @intCast(num_layers),
            .num_attention_heads = @intCast(num_heads),
            .num_key_value_heads = @intCast(num_kv_heads),
            .intermediate_size = @intCast(intermediate),
            .vocab_size = @intCast(vocab),
            .max_position_embeddings = @intCast(max_pos),
            .rms_norm_eps = if (cfg.getFloat("rms_norm_eps")) |v| @floatCast(v) else 1e-5,
            .rope_theta = if (cfg.getFloat("rope_theta")) |v| @floatCast(v) else 10000.0,
            .head_dim = if (head_dim_field) |v| @intCast(v) else @intCast(@divExact(hidden_size, num_heads)),
            .tie_word_embeddings = cfg.getBool("tie_word_embeddings") orelse true,
        };
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "parse generic config.json" {
    const json =
        \\{"hidden_size": 4096, "model_type": "llama", "rope_theta": 500000.0}
    ;
    var cfg = try ConfigJson.initFromSlice(testing.allocator, json);
    defer cfg.deinit();

    try testing.expectEqual(@as(?i64, 4096), cfg.getInt("hidden_size"));
    try testing.expectEqualStrings("llama", cfg.getString("model_type").?);
    try testing.expectEqual(@as(?f64, 500000.0), cfg.getFloat("rope_theta"));
    try testing.expectEqual(@as(?bool, null), cfg.getBool("missing"));
}

test "LlamaConfig.fromConfig — Llama-3 8B shape" {
    const json =
        \\{
        \\  "hidden_size": 4096,
        \\  "num_hidden_layers": 32,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 8,
        \\  "intermediate_size": 14336,
        \\  "vocab_size": 128256,
        \\  "max_position_embeddings": 8192,
        \\  "rms_norm_eps": 1e-5,
        \\  "rope_theta": 500000.0,
        \\  "tie_word_embeddings": false
        \\}
    ;
    var cfg = try ConfigJson.initFromSlice(testing.allocator, json);
    defer cfg.deinit();

    const llama = try LlamaConfig.fromConfig(&cfg);
    try testing.expectEqual(@as(usize, 4096), llama.hidden_size);
    try testing.expectEqual(@as(usize, 32), llama.num_hidden_layers);
    try testing.expectEqual(@as(usize, 8), llama.num_key_value_heads);
    try testing.expectEqual(@as(usize, 128), llama.head_dim);
    try testing.expectEqual(false, llama.tie_word_embeddings);
}

test "LlamaConfig.fromConfig — InkubaLM 0.4B (MHA, no kv_heads field)" {
    // InkubaLM uses MHA so num_key_value_heads is absent → falls back to num_attention_heads.
    const json =
        \\{
        \\  "hidden_size": 2048,
        \\  "num_hidden_layers": 8,
        \\  "num_attention_heads": 32,
        \\  "intermediate_size": 5632,
        \\  "vocab_size": 61788,
        \\  "max_position_embeddings": 2048
        \\}
    ;
    var cfg = try ConfigJson.initFromSlice(testing.allocator, json);
    defer cfg.deinit();

    const llama = try LlamaConfig.fromConfig(&cfg);
    try testing.expectEqual(@as(usize, 32), llama.num_key_value_heads);
    try testing.expectEqual(@as(usize, 64), llama.head_dim);
    try testing.expectEqual(true, llama.tie_word_embeddings); // default
}
