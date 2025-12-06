// src/models/whisper/cache.zig
// KV Cache for efficient autoregressive decoding
//
// The KV cache stores key and value projections from previous tokens,
// avoiding redundant computation during generation. This reduces the
// decoder from O(n²) to O(n) complexity per token.
//
// For Whisper decoder:
// - Self-attention: Cache K/V for all previous decoder tokens
// - Cross-attention: Cache K/V for encoder output (computed once, reused)

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

/// KV cache for a single attention layer
pub const LayerKVCache = struct {
    /// Cached keys: [max_seq_len, hidden_dim]
    k_cache: Tensor,
    /// Cached values: [max_seq_len, hidden_dim]
    v_cache: Tensor,
    /// Current position (number of cached tokens)
    position: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Hidden dimension
    hidden_dim: usize,

    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, max_seq_len: usize, hidden_dim: usize) !Self {
        var k_shape = [_]usize{ max_seq_len, hidden_dim };
        var k_cache = try Tensor.init(allocator, &k_shape);
        errdefer k_cache.deinit();

        var v_shape = [_]usize{ max_seq_len, hidden_dim };
        var v_cache = try Tensor.init(allocator, &v_shape);

        return Self{
            .k_cache = k_cache,
            .v_cache = v_cache,
            .position = 0,
            .max_seq_len = max_seq_len,
            .hidden_dim = hidden_dim,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.k_cache.deinit();
        self.v_cache.deinit();
    }

    /// Append new key/value to cache and return full cached K/V
    /// k_new: [new_len, hidden_dim] - new keys to append
    /// v_new: [new_len, hidden_dim] - new values to append
    pub fn append(self: *Self, k_new: *const Tensor, v_new: *const Tensor) void {
        const new_len = k_new.shape[0];
        const hidden = self.hidden_dim;

        // Copy new keys into cache
        const k_start = self.position * hidden;
        @memcpy(
            self.k_cache.data[k_start .. k_start + new_len * hidden],
            k_new.data[0 .. new_len * hidden],
        );

        // Copy new values into cache
        const v_start = self.position * hidden;
        @memcpy(
            self.v_cache.data[v_start .. v_start + new_len * hidden],
            v_new.data[0 .. new_len * hidden],
        );

        self.position += new_len;
    }

    /// Get cached K tensor up to current position
    /// Returns a view (does not allocate) with shape [position, hidden_dim]
    pub fn getK(self: *const Self) Tensor {
        return Tensor{
            .data = self.k_cache.data[0 .. self.position * self.hidden_dim],
            .shape = &[_]usize{ self.position, self.hidden_dim },
            .allocator = null, // View, don't free
        };
    }

    /// Get cached V tensor up to current position
    pub fn getV(self: *const Self) Tensor {
        return Tensor{
            .data = self.v_cache.data[0 .. self.position * self.hidden_dim],
            .shape = &[_]usize{ self.position, self.hidden_dim },
            .allocator = null, // View, don't free
        };
    }

    /// Reset cache for new sequence
    pub fn reset(self: *Self) void {
        self.position = 0;
    }
};

/// Complete KV cache for Whisper decoder
/// Contains caches for all layers' self-attention and cross-attention
pub const DecoderKVCache = struct {
    /// Self-attention caches for each decoder layer
    self_attn_caches: []LayerKVCache,
    /// Cross-attention caches for each decoder layer (encoder K/V, computed once)
    cross_attn_caches: []LayerKVCache,
    /// Number of layers
    n_layers: usize,

    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        n_layers: usize,
        max_text_ctx: usize,
        max_audio_ctx: usize,
        hidden_dim: usize,
    ) !Self {
        var self_attn_caches = try allocator.alloc(LayerKVCache, n_layers);
        errdefer allocator.free(self_attn_caches);

        var self_init_count: usize = 0;
        errdefer {
            for (self_attn_caches[0..self_init_count]) |*c| c.deinit();
        }

        for (0..n_layers) |i| {
            self_attn_caches[i] = try LayerKVCache.init(allocator, max_text_ctx, hidden_dim);
            self_init_count += 1;
        }

        var cross_attn_caches = try allocator.alloc(LayerKVCache, n_layers);
        errdefer allocator.free(cross_attn_caches);

        var cross_init_count: usize = 0;
        errdefer {
            for (cross_attn_caches[0..cross_init_count]) |*c| c.deinit();
        }

        for (0..n_layers) |i| {
            cross_attn_caches[i] = try LayerKVCache.init(allocator, max_audio_ctx, hidden_dim);
            cross_init_count += 1;
        }

        return Self{
            .self_attn_caches = self_attn_caches,
            .cross_attn_caches = cross_attn_caches,
            .n_layers = n_layers,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.self_attn_caches) |*c| c.deinit();
        self.allocator.free(self.self_attn_caches);
        for (self.cross_attn_caches) |*c| c.deinit();
        self.allocator.free(self.cross_attn_caches);
    }

    /// Reset self-attention caches for new sequence
    /// (Cross-attention caches persist as encoder output doesn't change)
    pub fn resetSelfAttention(self: *Self) void {
        for (self.self_attn_caches) |*c| c.reset();
    }

    /// Reset all caches
    pub fn resetAll(self: *Self) void {
        for (self.self_attn_caches) |*c| c.reset();
        for (self.cross_attn_caches) |*c| c.reset();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "LayerKVCache init and append" {
    const allocator = std.testing.allocator;

    var cache = try LayerKVCache.init(allocator, 10, 4);
    defer cache.deinit();

    try std.testing.expectEqual(@as(usize, 0), cache.position);

    // Create some test data
    var k_shape = [_]usize{ 2, 4 };
    var k_new = try Tensor.init(allocator, &k_shape);
    defer k_new.deinit();
    for (k_new.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));

    var v_new = try Tensor.init(allocator, &k_shape);
    defer v_new.deinit();
    for (v_new.data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 10));

    // Append
    cache.append(&k_new, &v_new);
    try std.testing.expectEqual(@as(usize, 2), cache.position);

    // Get cached tensors
    const k_cached = cache.getK();
    const v_cached = cache.getV();

    try std.testing.expectEqual(@as(usize, 2), k_cached.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), k_cached.shape[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), k_cached.data[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), v_cached.data[0], 0.001);
}

test "DecoderKVCache init" {
    const allocator = std.testing.allocator;

    var cache = try DecoderKVCache.init(allocator, 4, 448, 1500, 384);
    defer cache.deinit();

    try std.testing.expectEqual(@as(usize, 4), cache.n_layers);
}
