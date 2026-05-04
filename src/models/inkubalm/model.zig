// src/models/inkubalm/model.zig
//
// Thin wrapper around arch/llama.zig + format/llama_loader.zig for
// InkubaLM-0.4B. There is intentionally no model-specific architecture
// code here — InkubaLM is a stock Llama. Everything specific lives in
// config.zig (the LlamaConfig literal) and the future tokenizer.

const std = @import("std");
const llama = @import("llama.zig");
const llama_loader = @import("llama_loader.zig");
const safetensors = @import("safetensors.zig");
const cfg_mod = @import("config.zig");

pub const config = cfg_mod.inkubalm_0_4b;

/// A loaded InkubaLM ready for inference. Owns its weights, KV cache,
/// and per-token scratch buffers.
pub const InkubaLM = struct {
    loaded: llama_loader.LoadedLlama,
    cache: llama.LlamaCache,
    scratch: llama.Scratch,
    file: safetensors.SafetensorsFile,
    allocator: std.mem.Allocator,

    pub fn loadFromFile(
        allocator: std.mem.Allocator,
        safetensors_path: []const u8,
    ) !InkubaLM {
        var file = try safetensors.SafetensorsFile.init(allocator, safetensors_path);
        errdefer file.deinit();

        var loaded = try llama_loader.loadFromSafetensors(allocator, config, &file);
        errdefer loaded.deinit();

        var cache = try llama.LlamaCache.init(allocator, config, config.max_seq_len);
        errdefer cache.deinit();

        const scratch = try llama.Scratch.init(allocator, config, config.max_seq_len);

        return .{
            .loaded = loaded,
            .cache = cache,
            .scratch = scratch,
            .file = file,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *InkubaLM) void {
        self.scratch.deinit();
        self.cache.deinit();
        self.loaded.deinit();
        self.file.deinit();
    }

    pub fn reset(self: *InkubaLM) void {
        self.cache.reset();
    }

    /// Run one forward pass for `token_id` at the given absolute position.
    /// Returns the next position. Logits live in `self.scratch.logits` until
    /// the next call.
    pub fn step(self: *InkubaLM, token_id: u32, position: usize) usize {
        llama.forwardToken(config, &self.loaded.weights, &self.cache, &self.scratch, token_id, position);
        return position + 1;
    }

    pub fn logits(self: *const InkubaLM) []const f32 {
        return self.scratch.logits;
    }

    /// Greedy generation: feed `prompt_ids`, then repeatedly take argmax.
    /// Stops at `max_new_tokens` or when `eos_id` is produced.
    /// Caller owns the returned slice.
    pub fn generateGreedy(
        self: *InkubaLM,
        allocator: std.mem.Allocator,
        prompt_ids: []const u32,
        max_new_tokens: usize,
        eos_id: ?u32,
    ) ![]u32 {
        self.reset();

        var out: std.ArrayList(u32) = .{};
        errdefer out.deinit(allocator);

        var position: usize = 0;

        // Prefill: drive every prompt token through the cache.
        for (prompt_ids) |tid| {
            position = self.step(tid, position);
        }

        // The logits for the *last* prompt token predict the first new token.
        var next: u32 = argmaxU32(self.logits());

        var produced: usize = 0;
        while (produced < max_new_tokens) : (produced += 1) {
            try out.append(allocator, next);
            if (eos_id) |e| if (next == e) break;
            if (position >= config.max_seq_len) break;
            position = self.step(next, position);
            next = argmaxU32(self.logits());
        }

        return out.toOwnedSlice(allocator);
    }
};

fn argmaxU32(values: []const f32) u32 {
    std.debug.assert(values.len > 0);
    var best: usize = 0;
    var best_v: f32 = values[0];
    for (values[1..], 1..) |v, i| {
        if (v > best_v) {
            best_v = v;
            best = i;
        }
    }
    return @intCast(best);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// We can't load real InkubaLM weights in unit tests (the file is 800 MB).
// What we CAN do: confirm the comptime config parses, the InkubaLM struct
// composes, and the helpers compile. End-to-end correctness will be
// covered by the Python validation harness.

const testing = std.testing;

test "InkubaLM config sanity" {
    try testing.expectEqual(@as(usize, 2048), config.hidden_size);
    try testing.expectEqual(@as(usize, 8), config.num_layers);
    try testing.expectEqual(@as(usize, 32), config.num_heads);
    try testing.expectEqual(@as(usize, 32), config.num_kv_heads);
    try testing.expectEqual(@as(usize, 64), config.head_dim);
    try testing.expectEqual(@as(usize, 61788), config.vocab_size);
    try testing.expectEqual(true, config.tie_word_embeddings);

    // Sanity: q_dim and kv_dim agree because MHA.
    try testing.expectEqual(config.qDim(), config.kvDim());
    try testing.expectEqual(@as(usize, 1), config.headsPerKv());
}

test "argmaxU32" {
    const xs = [_]f32{ 0.1, 0.5, -1.0, 0.9, 0.2 };
    try testing.expectEqual(@as(u32, 3), argmaxU32(&xs));
}
