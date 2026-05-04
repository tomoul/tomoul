// src/format/llama_loader.zig
//
// Bridge: safetensors file → arch.LlamaWeights.
//
// Knows the HuggingFace Llama naming convention:
//
//   model.embed_tokens.weight                                    [vocab, hidden]
//   model.norm.weight                                            [hidden]
//   lm_head.weight                                               [vocab, hidden]   (omitted if tied)
//   model.layers.{i}.input_layernorm.weight                      [hidden]
//   model.layers.{i}.post_attention_layernorm.weight             [hidden]
//   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight             [out, in]
//   model.layers.{i}.mlp.{gate,up,down}_proj.weight              [out, in]
//
// Used by Llama-3, Mistral, InkubaLM, N-ATLaS, AfroLlama, and most
// HF-format Llama derivatives.

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const llama = @import("llama.zig");
const safetensors = @import("safetensors.zig");

const LlamaConfig = llama.LlamaConfig;
const LlamaWeights = llama.LlamaWeights;
const LlamaLayerWeights = llama.LlamaLayerWeights;
const ProjectionWeight = llama.ProjectionWeight;
const SafetensorsFile = safetensors.SafetensorsFile;

pub const Error = error{
    InvalidConfig,
} || safetensors.Error || std.mem.Allocator.Error;

/// Bundle that owns every Tensor backing a loaded LlamaWeights.
/// Call `deinit` when done.
pub const LoadedLlama = struct {
    weights: LlamaWeights,
    /// All Tensors we allocated, in load order. We free them all at once.
    owned: std.ArrayList(Tensor),
    allocator: std.mem.Allocator,

    pub fn deinit(self: *LoadedLlama) void {
        for (self.owned.items) |*t| t.deinit();
        self.owned.deinit(self.allocator);
        self.allocator.free(self.weights.layers);
    }
};

/// Build an owned f32 Tensor by reading a safetensors entry, verifying its
/// shape, and wrapping the resulting slice. The Tensor takes ownership of
/// the slice's storage.
fn loadTensor(
    allocator: std.mem.Allocator,
    file: *const SafetensorsFile,
    name: []const u8,
    expected_shape: []const usize,
) !Tensor {
    const data = try file.readF32Checked(allocator, name, expected_shape);
    errdefer allocator.free(data);

    const owned_shape = try allocator.alloc(usize, expected_shape.len);
    errdefer allocator.free(owned_shape);
    @memcpy(owned_shape, expected_shape);

    return Tensor{
        .data = data,
        .shape = owned_shape,
        .allocator = allocator,
    };
}

fn loadProj(
    allocator: std.mem.Allocator,
    file: *const SafetensorsFile,
    name: []const u8,
    rows: usize,
    cols: usize,
    owned_list: *std.ArrayList(Tensor),
) !ProjectionWeight {
    const t = try loadTensor(allocator, file, name, &[_]usize{ rows, cols });
    try owned_list.append(allocator, t);
    return ProjectionWeight{ .f32 = t };
}

fn loadVec(
    allocator: std.mem.Allocator,
    file: *const SafetensorsFile,
    name: []const u8,
    dim: usize,
    owned_list: *std.ArrayList(Tensor),
) !Tensor {
    const t = try loadTensor(allocator, file, name, &[_]usize{dim});
    try owned_list.append(allocator, t);
    return t;
}

/// Load a Llama-family model from one safetensors file.
/// `cfg.tie_word_embeddings == true` reuses `embed_tokens` for `lm_head`
/// without an extra read.
pub fn loadFromSafetensors(
    allocator: std.mem.Allocator,
    cfg: LlamaConfig,
    file: *const SafetensorsFile,
) !LoadedLlama {
    if (cfg.num_heads % cfg.num_kv_heads != 0) return Error.InvalidConfig;

    var owned: std.ArrayList(Tensor) = .{};
    errdefer {
        for (owned.items) |*t| t.deinit();
        owned.deinit(allocator);
    }

    const hidden = cfg.hidden_size;
    const q_dim = cfg.qDim();
    const kv_dim = cfg.kvDim();

    const embed_t = try loadTensor(
        allocator,
        file,
        "model.embed_tokens.weight",
        &[_]usize{ cfg.vocab_size, hidden },
    );
    try owned.append(allocator, embed_t);
    const embed_w = ProjectionWeight{ .f32 = embed_t };

    const final_norm = try loadVec(allocator, file, "model.norm.weight", hidden, &owned);

    const lm_head: ProjectionWeight = if (cfg.tie_word_embeddings)
        embed_w
    else blk: {
        const head_t = try loadTensor(
            allocator,
            file,
            "lm_head.weight",
            &[_]usize{ cfg.vocab_size, hidden },
        );
        try owned.append(allocator, head_t);
        break :blk ProjectionWeight{ .f32 = head_t };
    };

    var layers = try allocator.alloc(LlamaLayerWeights, cfg.num_layers);
    errdefer allocator.free(layers);

    var name_buf: [128]u8 = undefined;

    for (0..cfg.num_layers) |i| {
        const fmt = struct {
            fn n(buf: []u8, comptime tmpl: []const u8, idx: usize) ![]const u8 {
                return std.fmt.bufPrint(buf, tmpl, .{idx});
            }
        };

        const in_ln_name = try fmt.n(&name_buf, "model.layers.{d}.input_layernorm.weight", i);
        const in_ln = try loadVec(allocator, file, in_ln_name, hidden, &owned);

        const post_ln_name = try fmt.n(&name_buf, "model.layers.{d}.post_attention_layernorm.weight", i);
        const post_ln = try loadVec(allocator, file, post_ln_name, hidden, &owned);

        const q_name = try fmt.n(&name_buf, "model.layers.{d}.self_attn.q_proj.weight", i);
        const q_p = try loadProj(allocator, file, q_name, q_dim, hidden, &owned);

        const k_name = try fmt.n(&name_buf, "model.layers.{d}.self_attn.k_proj.weight", i);
        const k_p = try loadProj(allocator, file, k_name, kv_dim, hidden, &owned);

        const v_name = try fmt.n(&name_buf, "model.layers.{d}.self_attn.v_proj.weight", i);
        const v_p = try loadProj(allocator, file, v_name, kv_dim, hidden, &owned);

        const o_name = try fmt.n(&name_buf, "model.layers.{d}.self_attn.o_proj.weight", i);
        const o_p = try loadProj(allocator, file, o_name, hidden, q_dim, &owned);

        const g_name = try fmt.n(&name_buf, "model.layers.{d}.mlp.gate_proj.weight", i);
        const g_p = try loadProj(allocator, file, g_name, cfg.intermediate_size, hidden, &owned);

        const u_name = try fmt.n(&name_buf, "model.layers.{d}.mlp.up_proj.weight", i);
        const u_p = try loadProj(allocator, file, u_name, cfg.intermediate_size, hidden, &owned);

        const d_name = try fmt.n(&name_buf, "model.layers.{d}.mlp.down_proj.weight", i);
        const d_p = try loadProj(allocator, file, d_name, hidden, cfg.intermediate_size, &owned);

        layers[i] = .{
            .input_layernorm = in_ln,
            .post_attn_layernorm = post_ln,
            .q_proj = q_p,
            .k_proj = k_p,
            .v_proj = v_p,
            .o_proj = o_p,
            .gate_proj = g_p,
            .up_proj = u_p,
            .down_proj = d_p,
        };
    }

    return LoadedLlama{
        .weights = .{
            .embed_tokens = embed_w,
            .final_norm = final_norm,
            .lm_head = lm_head,
            .layers = layers,
            .allocator = allocator,
        },
        .owned = owned,
        .allocator = allocator,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Build a synthetic safetensors blob with all weights for a 1-layer Llama.
/// Every tensor is filled with deterministic values from `seed`.
fn buildLlamaBlob(
    allocator: std.mem.Allocator,
    cfg: LlamaConfig,
    seed: u64,
) ![]u8 {
    std.debug.assert(cfg.num_layers == 1);

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    const hidden = cfg.hidden_size;
    const q_dim = cfg.qDim();
    const kv_dim = cfg.kvDim();

    // Build header JSON and parallel data blocks.
    var header_buf: std.ArrayList(u8) = .{};
    defer header_buf.deinit(allocator);
    const hw = header_buf.writer(allocator);

    var data_buf: std.ArrayList(u8) = .{};
    defer data_buf.deinit(allocator);

    var first = true;
    try hw.writeAll("{");

    const Helper = struct {
        fn appendTensor(
            a: std.mem.Allocator,
            hwriter: anytype,
            dbuf: *std.ArrayList(u8),
            r: std.Random,
            f: *bool,
            name: []const u8,
            shape: []const usize,
        ) !void {
            if (!f.*) try hwriter.writeAll(",");
            f.* = false;

            var elem_count: usize = 1;
            for (shape) |d| elem_count *= d;
            const start = dbuf.items.len;
            const end = start + elem_count * 4;

            try hwriter.print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{name});
            for (shape, 0..) |d, i| {
                if (i > 0) try hwriter.writeAll(",");
                try hwriter.print("{d}", .{d});
            }
            try hwriter.print("],\"data_offsets\":[{d},{d}]}}", .{ start, end });

            // Append elem_count random f32 values as raw bytes.
            const old_len = dbuf.items.len;
            try dbuf.resize(a, old_len + elem_count * 4);
            for (0..elem_count) |i| {
                const v = (r.float(f32) - 0.5) * 0.2;
                const bytes: [4]u8 = @bitCast(v);
                @memcpy(dbuf.items[old_len + i * 4 ..][0..4], &bytes);
            }
        }
    };

    try Helper.appendTensor(allocator, hw, &data_buf, rng, &first, "model.embed_tokens.weight", &[_]usize{ cfg.vocab_size, hidden });
    try Helper.appendTensor(allocator, hw, &data_buf, rng, &first, "model.norm.weight", &[_]usize{hidden});
    try Helper.appendTensor(allocator, hw, &data_buf, rng, &first, "model.layers.0.input_layernorm.weight", &[_]usize{hidden});
    try Helper.appendTensor(allocator, hw, &data_buf, rng, &first, "model.layers.0.post_attention_layernorm.weight", &[_]usize{hidden});
    try Helper.appendTensor(allocator, hw, &data_buf, rng, &first, "model.layers.0.self_attn.q_proj.weight", &[_]usize{ q_dim, hidden });
    try Helper.appendTensor(allocator, hw, &data_buf, rng, &first, "model.layers.0.self_attn.k_proj.weight", &[_]usize{ kv_dim, hidden });
    try Helper.appendTensor(allocator, hw, &data_buf, rng, &first, "model.layers.0.self_attn.v_proj.weight", &[_]usize{ kv_dim, hidden });
    try Helper.appendTensor(allocator, hw, &data_buf, rng, &first, "model.layers.0.self_attn.o_proj.weight", &[_]usize{ hidden, q_dim });
    try Helper.appendTensor(allocator, hw, &data_buf, rng, &first, "model.layers.0.mlp.gate_proj.weight", &[_]usize{ cfg.intermediate_size, hidden });
    try Helper.appendTensor(allocator, hw, &data_buf, rng, &first, "model.layers.0.mlp.up_proj.weight", &[_]usize{ cfg.intermediate_size, hidden });
    try Helper.appendTensor(allocator, hw, &data_buf, rng, &first, "model.layers.0.mlp.down_proj.weight", &[_]usize{ hidden, cfg.intermediate_size });
    try hw.writeAll("}");

    const header_bytes = header_buf.items;
    const total = 8 + header_bytes.len + data_buf.items.len;
    const out = try allocator.alloc(u8, total);
    std.mem.writeInt(u64, out[0..8], header_bytes.len, .little);
    @memcpy(out[8..][0..header_bytes.len], header_bytes);
    @memcpy(out[8 + header_bytes.len ..][0..data_buf.items.len], data_buf.items);
    return out;
}

test "loadFromSafetensors: end-to-end synthetic blob → forwardToken" {
    const allocator = testing.allocator;

    const cfg = LlamaConfig{
        .hidden_size = 16,
        .num_layers = 1,
        .num_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 4,
        .intermediate_size = 32,
        .vocab_size = 24,
        .max_seq_len = 8,
        .rms_norm_eps = 1e-5,
        .rope_theta = 10000.0,
        .tie_word_embeddings = true,
    };

    const blob = try buildLlamaBlob(allocator, cfg, 0xDEAD_BEEF);
    defer allocator.free(blob);

    var file = try SafetensorsFile.initFromBytes(allocator, blob);
    defer file.deinit();

    var loaded = try loadFromSafetensors(allocator, cfg, &file);
    defer loaded.deinit();

    try testing.expectEqual(cfg.num_layers, loaded.weights.layers.len);

    var cache = try llama.LlamaCache.init(allocator, cfg, cfg.max_seq_len);
    defer cache.deinit();

    var scratch = try llama.Scratch.init(allocator, cfg, cfg.max_seq_len);
    defer scratch.deinit();

    llama.forwardToken(cfg, &loaded.weights, &cache, &scratch, 0, 0);
    llama.forwardToken(cfg, &loaded.weights, &cache, &scratch, 1, 1);

    try testing.expectEqual(@as(usize, 2), cache.layers[0].length);
    for (scratch.logits) |v| try testing.expect(std.math.isFinite(v));
}

test "loadFromSafetensors: missing tensor surfaces TensorNotFound" {
    const allocator = testing.allocator;

    // Header lists only the embedding. Loader should fail looking up norm.
    const header =
        \\{"model.embed_tokens.weight":{"dtype":"F32","shape":[2,4],"data_offsets":[0,32]}}
    ;
    const total = 8 + header.len + 32;
    const buf = try allocator.alloc(u8, total);
    defer allocator.free(buf);
    std.mem.writeInt(u64, buf[0..8], header.len, .little);
    @memcpy(buf[8..][0..header.len], header);
    @memset(buf[8 + header.len ..][0..32], 0);

    var file = try SafetensorsFile.initFromBytes(allocator, buf);
    defer file.deinit();

    const cfg = LlamaConfig{
        .hidden_size = 4,
        .num_layers = 1,
        .num_heads = 2,
        .num_kv_heads = 2,
        .head_dim = 2,
        .intermediate_size = 4,
        .vocab_size = 2,
        .max_seq_len = 4,
    };

    try testing.expectError(
        safetensors.Error.TensorNotFound,
        loadFromSafetensors(allocator, cfg, &file),
    );
}
