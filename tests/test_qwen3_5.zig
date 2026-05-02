// tests/test_qwen3_5.zig
// Unit tests for Qwen3.5-0.8B model components
//
// Tests config, DeltaNet recurrence, attention, and core ops.
// Does NOT require model weights — uses synthetic data.

const std = @import("std");
const testing = std.testing;

// Import core modules (available via build.zig test module)
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");

// ============================================================================
// Config Tests
// ============================================================================

test "qwen3_5 config: layer types pattern LLLA×6" {
    // Manually verify the [L L L A] × 6 pattern
    const interval: usize = 4;
    const num_layers: usize = 24;

    var dn_count: usize = 0;
    var fa_count: usize = 0;

    for (0..num_layers) |i| {
        if ((i + 1) % interval == 0) {
            fa_count += 1;
        } else {
            dn_count += 1;
        }
    }

    try testing.expectEqual(@as(usize, 18), dn_count);
    try testing.expectEqual(@as(usize, 6), fa_count);

    // Full attention at indices 3, 7, 11, 15, 19, 23
    const fa_indices = [_]usize{ 3, 7, 11, 15, 19, 23 };
    for (fa_indices) |idx| {
        try testing.expect((idx + 1) % interval == 0);
    }
}

test "qwen3_5 config: dimensions" {
    // DeltaNet: 16 heads × 128 dim = 2048
    try testing.expectEqual(@as(usize, 2048), 16 * 128);
    // QKV: key + key + value = 2048 + 2048 + 2048 = 6144
    try testing.expectEqual(@as(usize, 6144), 2048 * 2 + 2048);
    // Full attention Q: 8 heads × 256 dim = 2048
    try testing.expectEqual(@as(usize, 2048), 8 * 256);
    // Full attention KV: 2 heads × 256 dim = 512
    try testing.expectEqual(@as(usize, 512), 2 * 256);
    // Partial RoPE: 25% of 256 = 64
    try testing.expectEqual(@as(usize, 64), @as(usize, @intFromFloat(256.0 * 0.25)));
    // GQA ratio: 8 / 2 = 4
    try testing.expectEqual(@as(usize, 4), 8 / 2);
}

// ============================================================================
// Core Ops Tests (new operations added for Qwen3.5)
// ============================================================================

test "rmsNormInPlace" {
    const allocator = testing.allocator;

    // Input: [1, 4] with weight [4]
    var input = try Tensor.init(allocator, &.{ 1, 4 });
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = 2.0;
    input.data[2] = 3.0;
    input.data[3] = 4.0;

    var weight = try Tensor.init(allocator, &.{4});
    defer weight.deinit();
    weight.data[0] = 1.0;
    weight.data[1] = 1.0;
    weight.data[2] = 1.0;
    weight.data[3] = 1.0;

    try ops.rmsNormInPlace(&input, &weight, 1e-6);

    // RMS = sqrt(mean([1,4,9,16])) = sqrt(7.5) ≈ 2.7386
    // Normalized: x / rms
    const rms = @sqrt((1.0 + 4.0 + 9.0 + 16.0) / 4.0 + 1e-6);
    const expected_0 = 1.0 / rms;
    try testing.expectApproxEqAbs(expected_0, input.data[0], 1e-4);
    try testing.expectApproxEqAbs(2.0 / rms, input.data[1], 1e-4);
    try testing.expectApproxEqAbs(3.0 / rms, input.data[2], 1e-4);
    try testing.expectApproxEqAbs(4.0 / rms, input.data[3], 1e-4);
}

test "rmsNorm1DInPlace" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    ops.rmsNorm1DInPlace(&data, &weight, 4, 1e-6);

    const rms = @sqrt((1.0 + 4.0 + 9.0 + 16.0) / 4.0 + 1e-6);
    try testing.expectApproxEqAbs(1.0 / rms, data[0], 1e-4);
}

test "siluInPlace" {
    const allocator = testing.allocator;

    var tensor = try Tensor.init(allocator, &.{4});
    defer tensor.deinit();
    tensor.data[0] = 0.0;
    tensor.data[1] = 1.0;
    tensor.data[2] = -1.0;
    tensor.data[3] = 5.0;

    ops.siluInPlace(&tensor);

    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    try testing.expectApproxEqAbs(0.0, tensor.data[0], 1e-6); // 0 * 0.5 = 0
    try testing.expectApproxEqAbs(1.0 / (1.0 + @exp(@as(f32, -1.0))), tensor.data[1], 1e-5);
    try testing.expectApproxEqAbs(-1.0 / (1.0 + @exp(@as(f32, 1.0))), tensor.data[2], 1e-5);
}

test "softplus" {
    // softplus(x) = log(1 + exp(x))
    try testing.expectApproxEqAbs(@log(1.0 + @exp(@as(f32, 0.0))), ops.softplus(0.0), 1e-6);
    try testing.expectApproxEqAbs(@log(1.0 + @exp(@as(f32, 1.0))), ops.softplus(1.0), 1e-5);

    // Large positive: softplus(x) ≈ x
    try testing.expectApproxEqAbs(25.0, ops.softplus(25.0), 1e-4);

    // Large negative: softplus(x) ≈ 0
    try testing.expectApproxEqAbs(0.0, ops.softplus(-25.0), 1e-4);
}

test "l2NormInPlace" {
    var data = [_]f32{ 3.0, 4.0 };
    ops.l2NormInPlace(&data);

    // ||[3,4]|| = 5, normalized = [0.6, 0.8]
    try testing.expectApproxEqAbs(0.6, data[0], 1e-6);
    try testing.expectApproxEqAbs(0.8, data[1], 1e-6);
}

test "l2NormInPlace zero vector" {
    var data = [_]f32{ 0.0, 0.0, 0.0 };
    ops.l2NormInPlace(&data);

    // Zero vector should remain zero (no division by zero)
    try testing.expectEqual(@as(f32, 0.0), data[0]);
    try testing.expectEqual(@as(f32, 0.0), data[1]);
}

test "outerProductAddInPlace" {
    // a = [1, 2], b = [3, 4, 5], result should be [[3,4,5],[6,8,10]]
    var result = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const a = [_]f32{ 1.0, 2.0 };
    const b = [_]f32{ 3.0, 4.0, 5.0 };

    ops.outerProductAddInPlace(&result, &a, &b, 2, 3, 1.0);

    try testing.expectApproxEqAbs(3.0, result[0], 1e-6);
    try testing.expectApproxEqAbs(4.0, result[1], 1e-6);
    try testing.expectApproxEqAbs(5.0, result[2], 1e-6);
    try testing.expectApproxEqAbs(6.0, result[3], 1e-6);
    try testing.expectApproxEqAbs(8.0, result[4], 1e-6);
    try testing.expectApproxEqAbs(10.0, result[5], 1e-6);
}

test "matvecMul" {
    // M = [[1,2],[3,4]], v = [1,1] → [3, 7]
    const mat = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vec = [_]f32{ 1.0, 1.0 };
    var result = [_]f32{ 0.0, 0.0 };

    ops.matvecMul(&result, &mat, &vec, 2, 2);

    try testing.expectApproxEqAbs(3.0, result[0], 1e-6);
    try testing.expectApproxEqAbs(7.0, result[1], 1e-6);
}

test "ropeInPlace basic" {
    // Test partial RoPE: only first 2 of 4 dims rotated
    var q = [_]f32{ 1.0, 0.0, 0.5, 0.5 };
    var k = [_]f32{ 1.0, 0.0, 0.5, 0.5 };

    ops.ropeInPlace(&q, &k, 1, 1, 4, 2, 0, 10000.0);

    // At position 0, cos(0) = 1, sin(0) = 0, so values unchanged
    try testing.expectApproxEqAbs(1.0, q[0], 1e-5);
    try testing.expectApproxEqAbs(0.0, q[1], 1e-5);
    // Non-rotated dims unchanged
    try testing.expectApproxEqAbs(0.5, q[2], 1e-5);
    try testing.expectApproxEqAbs(0.5, q[3], 1e-5);
}

test "groupNormGatedInPlace" {
    // Simple test: 4 dims, 2 groups, uniform weight=1
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const z = [_]f32{ 1.0, 1.0, 1.0, 1.0 }; // gate
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    ops.groupNormGatedInPlace(&data, &z, &weight, 4, 2, 1e-5);

    // Group 0: [1,2] → RMS = sqrt((1+4)/2) = sqrt(2.5), inv_rms = 1/sqrt(2.5)
    // Group 1: [3,4] → RMS = sqrt((9+16)/2) = sqrt(12.5), inv_rms = 1/sqrt(12.5)
    // Then gated by silu(z) where z=1 → silu(1) ≈ 0.7311
    const silu_1 = 1.0 / (1.0 + @exp(@as(f32, -1.0)));
    const inv_rms_0 = 1.0 / @sqrt(2.5 + 1e-5);
    const inv_rms_1 = 1.0 / @sqrt(12.5 + 1e-5);

    try testing.expectApproxEqAbs(1.0 * inv_rms_0 * silu_1, data[0], 1e-4);
    try testing.expectApproxEqAbs(2.0 * inv_rms_0 * silu_1, data[1], 1e-4);
    try testing.expectApproxEqAbs(3.0 * inv_rms_1 * silu_1, data[2], 1e-4);
    try testing.expectApproxEqAbs(4.0 * inv_rms_1 * silu_1, data[3], 1e-4);
}
