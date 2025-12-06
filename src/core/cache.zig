// src/core/cache.zig
// KV Cache for efficient autoregressive decoding in Whisper and other decoder models
//
// Without KV caching, generating N tokens requires O(N²) compute because we
// recompute all previous key/value projections at each step. With caching,
// we only compute K,V for the new token and append it to the cache, achieving O(N).
//
// Usage:
//   var cache = try KVCache.init(allocator, num_layers, max_seq_len, hidden_dim);
//   defer cache.deinit();
//
//   // During generation, for each new token:
//   cache.append(layer_idx, &new_k, &new_v);
//   const all_keys = cache.getKeys(layer_idx);
//   const all_values = cache.getValues(layer_idx);
//   // Use all_keys, all_values for attention with the new query

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

/// Error types for KV cache operations
pub const CacheError = error{
    CacheFull,
    InvalidLayer,
    OutOfMemory,
};

/// Key-Value cache for decoder self-attention
/// Stores K,V projections for all previous tokens to avoid recomputation
pub const KVCache = struct {
    allocator: std.mem.Allocator,

    /// Per-layer key cache: [num_layers] each is [max_length, hidden_dim]
    keys: []Tensor,

    /// Per-layer value cache: [num_layers] each is [max_length, hidden_dim]
    values: []Tensor,

    /// Number of tokens currently cached (same across all layers)
    current_length: usize,

    /// Maximum number of tokens the cache can hold
    max_length: usize,

    /// Number of transformer layers
    num_layers: usize,

    /// Hidden dimension (typically num_heads * head_dim)
    hidden_dim: usize,

    const Self = @This();

    /// Initialize a KV cache for a decoder
    /// Allocates memory for all layers upfront (max_length * hidden_dim * num_layers * 2)
    pub fn init(
        allocator: std.mem.Allocator,
        num_layers: usize,
        max_length: usize,
        hidden_dim: usize,
    ) !Self {
        var keys = try allocator.alloc(Tensor, num_layers);
        errdefer allocator.free(keys);

        var values = try allocator.alloc(Tensor, num_layers);
        errdefer {
            allocator.free(values);
        }

        var initialized_keys: usize = 0;
        var initialized_values: usize = 0;

        errdefer {
            for (keys[0..initialized_keys]) |*k| k.deinit();
            for (values[0..initialized_values]) |*v| v.deinit();
        }

        var shape = [_]usize{ max_length, hidden_dim };
        for (0..num_layers) |i| {
            keys[i] = try Tensor.init(allocator, &shape);
            initialized_keys += 1;
            values[i] = try Tensor.init(allocator, &shape);
            initialized_values += 1;
        }

        return .{
            .allocator = allocator,
            .keys = keys,
            .values = values,
            .current_length = 0,
            .max_length = max_length,
            .num_layers = num_layers,
            .hidden_dim = hidden_dim,
        };
    }

    /// Append new key/value tensors for a single token to the cache at the given layer
    /// k: [1, hidden_dim] - key for the new token
    /// v: [1, hidden_dim] - value for the new token
    pub fn append(self: *Self, layer: usize, k: *const Tensor, v: *const Tensor) CacheError!void {
        if (layer >= self.num_layers) {
            return CacheError.InvalidLayer;
        }
        if (self.current_length >= self.max_length) {
            return CacheError.CacheFull;
        }

        const offset = self.current_length * self.hidden_dim;

        // Copy key data to cache
        @memcpy(
            self.keys[layer].data[offset..][0..self.hidden_dim],
            k.data[0..self.hidden_dim],
        );

        // Copy value data to cache
        @memcpy(
            self.values[layer].data[offset..][0..self.hidden_dim],
            v.data[0..self.hidden_dim],
        );
    }

    /// Append new key/value for multiple tokens at once (e.g., prompt processing)
    /// k: [num_tokens, hidden_dim]
    /// v: [num_tokens, hidden_dim]
    pub fn appendBatch(self: *Self, layer: usize, k: *const Tensor, v: *const Tensor) CacheError!void {
        if (layer >= self.num_layers) {
            return CacheError.InvalidLayer;
        }

        const num_tokens = k.shape[0];
        if (self.current_length + num_tokens > self.max_length) {
            return CacheError.CacheFull;
        }

        const offset = self.current_length * self.hidden_dim;
        const copy_size = num_tokens * self.hidden_dim;

        @memcpy(
            self.keys[layer].data[offset..][0..copy_size],
            k.data[0..copy_size],
        );

        @memcpy(
            self.values[layer].data[offset..][0..copy_size],
            v.data[0..copy_size],
        );
    }

    /// Increment the current length after appending to all layers
    /// Call this once after appending K,V to all layers for a token
    pub fn incrementLength(self: *Self) void {
        self.current_length += 1;
    }

    /// Increment the current length by a batch size (after appendBatch)
    pub fn incrementLengthBy(self: *Self, count: usize) void {
        self.current_length += count;
    }

    /// Get keys for a layer up to the current cached length
    /// Returns: [current_length, hidden_dim] view into the cache
    pub fn getKeys(self: *const Self, layer: usize) []const f32 {
        const size = self.current_length * self.hidden_dim;
        return self.keys[layer].data[0..size];
    }

    /// Get values for a layer up to the current cached length
    /// Returns: [current_length, hidden_dim] view into the cache
    pub fn getValues(self: *const Self, layer: usize) []const f32 {
        const size = self.current_length * self.hidden_dim;
        return self.values[layer].data[0..size];
    }

    /// Get the current number of cached tokens
    pub fn getLength(self: *const Self) usize {
        return self.current_length;
    }

    /// Check if cache is full
    pub fn isFull(self: *const Self) bool {
        return self.current_length >= self.max_length;
    }

    /// Reset the cache for a new sequence
    /// Does not deallocate memory, just resets the length counter
    pub fn reset(self: *Self) void {
        self.current_length = 0;
    }

    /// Clean up all allocated memory
    pub fn deinit(self: *Self) void {
        for (self.keys) |*k| k.deinit();
        for (self.values) |*v| v.deinit();
        self.allocator.free(self.keys);
        self.allocator.free(self.values);
    }
};

/// Cross-attention KV cache
/// For encoder-decoder models, the encoder output is computed once and cached
/// The decoder's cross-attention uses this cache at every generation step
pub const CrossAttentionCache = struct {
    allocator: std.mem.Allocator,

    /// Per-layer key cache from encoder: [num_layers] each is [encoder_seq_len, hidden_dim]
    keys: []Tensor,

    /// Per-layer value cache from encoder: [num_layers] each is [encoder_seq_len, hidden_dim]
    values: []Tensor,

    /// Length of the encoder sequence (fixed after initialization)
    encoder_length: usize,

    /// Number of decoder layers that use cross-attention
    num_layers: usize,

    /// Hidden dimension
    hidden_dim: usize,

    /// Whether the cache has been populated
    is_populated: bool,

    const Self = @This();

    /// Initialize an empty cross-attention cache
    /// Will be populated once when the encoder output is available
    pub fn init(
        allocator: std.mem.Allocator,
        num_layers: usize,
        encoder_length: usize,
        hidden_dim: usize,
    ) !Self {
        var keys = try allocator.alloc(Tensor, num_layers);
        errdefer allocator.free(keys);

        var values = try allocator.alloc(Tensor, num_layers);
        errdefer allocator.free(values);

        var initialized_keys: usize = 0;
        var initialized_values: usize = 0;

        errdefer {
            for (keys[0..initialized_keys]) |*k| k.deinit();
            for (values[0..initialized_values]) |*v| v.deinit();
        }

        var shape = [_]usize{ encoder_length, hidden_dim };
        for (0..num_layers) |i| {
            keys[i] = try Tensor.init(allocator, &shape);
            initialized_keys += 1;
            values[i] = try Tensor.init(allocator, &shape);
            initialized_values += 1;
        }

        return .{
            .allocator = allocator,
            .keys = keys,
            .values = values,
            .encoder_length = encoder_length,
            .num_layers = num_layers,
            .hidden_dim = hidden_dim,
            .is_populated = false,
        };
    }

    /// Populate the cache with K,V projections from encoder output
    /// Called once after encoding the audio
    /// k: [encoder_length, hidden_dim]
    /// v: [encoder_length, hidden_dim]
    pub fn populate(self: *Self, layer: usize, k: *const Tensor, v: *const Tensor) void {
        const size = self.encoder_length * self.hidden_dim;
        @memcpy(self.keys[layer].data[0..size], k.data[0..size]);
        @memcpy(self.values[layer].data[0..size], v.data[0..size]);
    }

    /// Mark the cache as fully populated (call after populating all layers)
    pub fn markPopulated(self: *Self) void {
        self.is_populated = true;
    }

    /// Get keys for a layer (full encoder sequence)
    pub fn getKeys(self: *const Self, layer: usize) []const f32 {
        const size = self.encoder_length * self.hidden_dim;
        return self.keys[layer].data[0..size];
    }

    /// Get values for a layer (full encoder sequence)
    pub fn getValues(self: *const Self, layer: usize) []const f32 {
        const size = self.encoder_length * self.hidden_dim;
        return self.values[layer].data[0..size];
    }

    /// Check if the cache is ready to use
    pub fn isReady(self: *const Self) bool {
        return self.is_populated;
    }

    /// Reset the cache for a new audio input
    pub fn reset(self: *Self) void {
        self.is_populated = false;
    }

    /// Clean up all allocated memory
    pub fn deinit(self: *Self) void {
        for (self.keys) |*k| k.deinit();
        for (self.values) |*v| v.deinit();
        self.allocator.free(self.keys);
        self.allocator.free(self.values);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "KVCache init and deinit" {
    const allocator = std.testing.allocator;

    var cache = try KVCache.init(allocator, 4, 128, 64);
    defer cache.deinit();

    try std.testing.expectEqual(@as(usize, 4), cache.num_layers);
    try std.testing.expectEqual(@as(usize, 128), cache.max_length);
    try std.testing.expectEqual(@as(usize, 64), cache.hidden_dim);
    try std.testing.expectEqual(@as(usize, 0), cache.current_length);
}

test "KVCache append single token" {
    const allocator = std.testing.allocator;

    var cache = try KVCache.init(allocator, 2, 10, 4);
    defer cache.deinit();

    // Create K,V for a single token
    var kv_shape = [_]usize{ 1, 4 };
    var k = try Tensor.init(allocator, &kv_shape);
    defer k.deinit();
    var v = try Tensor.init(allocator, &kv_shape);
    defer v.deinit();

    k.data[0] = 1.0;
    k.data[1] = 2.0;
    k.data[2] = 3.0;
    k.data[3] = 4.0;

    v.data[0] = 5.0;
    v.data[1] = 6.0;
    v.data[2] = 7.0;
    v.data[3] = 8.0;

    // Append to layer 0
    try cache.append(0, &k, &v);
    cache.incrementLength();

    try std.testing.expectEqual(@as(usize, 1), cache.current_length);

    // Verify cached values
    const keys = cache.getKeys(0);
    try std.testing.expectEqual(@as(usize, 4), keys.len);
    try std.testing.expectEqual(@as(f32, 1.0), keys[0]);
    try std.testing.expectEqual(@as(f32, 4.0), keys[3]);

    const values = cache.getValues(0);
    try std.testing.expectEqual(@as(f32, 5.0), values[0]);
    try std.testing.expectEqual(@as(f32, 8.0), values[3]);
}

test "KVCache multiple tokens" {
    const allocator = std.testing.allocator;

    var cache = try KVCache.init(allocator, 1, 10, 2);
    defer cache.deinit();

    var kv_shape = [_]usize{ 1, 2 };

    // Append token 0
    var k0 = try Tensor.init(allocator, &kv_shape);
    defer k0.deinit();
    var v0 = try Tensor.init(allocator, &kv_shape);
    defer v0.deinit();
    k0.data[0] = 1.0;
    k0.data[1] = 2.0;
    v0.data[0] = 3.0;
    v0.data[1] = 4.0;

    try cache.append(0, &k0, &v0);
    cache.incrementLength();

    // Append token 1
    var k1 = try Tensor.init(allocator, &kv_shape);
    defer k1.deinit();
    var v1 = try Tensor.init(allocator, &kv_shape);
    defer v1.deinit();
    k1.data[0] = 5.0;
    k1.data[1] = 6.0;
    v1.data[0] = 7.0;
    v1.data[1] = 8.0;

    try cache.append(0, &k1, &v1);
    cache.incrementLength();

    try std.testing.expectEqual(@as(usize, 2), cache.current_length);

    // Should return both tokens
    const keys = cache.getKeys(0);
    try std.testing.expectEqual(@as(usize, 4), keys.len); // 2 tokens * 2 hidden
    try std.testing.expectEqual(@as(f32, 1.0), keys[0]); // token 0
    try std.testing.expectEqual(@as(f32, 2.0), keys[1]);
    try std.testing.expectEqual(@as(f32, 5.0), keys[2]); // token 1
    try std.testing.expectEqual(@as(f32, 6.0), keys[3]);
}

test "KVCache reset" {
    const allocator = std.testing.allocator;

    var cache = try KVCache.init(allocator, 1, 10, 2);
    defer cache.deinit();

    var kv_shape = [_]usize{ 1, 2 };
    var k = try Tensor.init(allocator, &kv_shape);
    defer k.deinit();
    var v = try Tensor.init(allocator, &kv_shape);
    defer v.deinit();

    try cache.append(0, &k, &v);
    cache.incrementLength();
    try std.testing.expectEqual(@as(usize, 1), cache.current_length);

    cache.reset();
    try std.testing.expectEqual(@as(usize, 0), cache.current_length);
    try std.testing.expectEqual(@as(usize, 0), cache.getKeys(0).len);
}

test "KVCache full error" {
    const allocator = std.testing.allocator;

    var cache = try KVCache.init(allocator, 1, 2, 2);
    defer cache.deinit();

    var kv_shape = [_]usize{ 1, 2 };
    var k = try Tensor.init(allocator, &kv_shape);
    defer k.deinit();
    var v = try Tensor.init(allocator, &kv_shape);
    defer v.deinit();

    // Fill cache to max
    try cache.append(0, &k, &v);
    cache.incrementLength();
    try cache.append(0, &k, &v);
    cache.incrementLength();

    try std.testing.expect(cache.isFull());

    // Third append should fail
    try std.testing.expectError(CacheError.CacheFull, cache.append(0, &k, &v));
}

test "CrossAttentionCache basic usage" {
    const allocator = std.testing.allocator;

    var cache = try CrossAttentionCache.init(allocator, 2, 5, 4);
    defer cache.deinit();

    try std.testing.expect(!cache.isReady());

    // Populate layer 0
    var kv_shape = [_]usize{ 5, 4 };
    var k = try Tensor.init(allocator, &kv_shape);
    defer k.deinit();
    var v = try Tensor.init(allocator, &kv_shape);
    defer v.deinit();

    for (k.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i));
    }
    for (v.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) + 100.0;
    }

    cache.populate(0, &k, &v);
    cache.populate(1, &k, &v);
    cache.markPopulated();

    try std.testing.expect(cache.isReady());

    // Verify values
    const keys = cache.getKeys(0);
    try std.testing.expectEqual(@as(usize, 20), keys.len); // 5 * 4
    try std.testing.expectEqual(@as(f32, 0.0), keys[0]);
    try std.testing.expectEqual(@as(f32, 19.0), keys[19]);

    const values = cache.getValues(0);
    try std.testing.expectEqual(@as(f32, 100.0), values[0]);
    try std.testing.expectEqual(@as(f32, 119.0), values[19]);
}
