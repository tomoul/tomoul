const std = @import("std");

// Core modules
pub const tensor = @import("core/tensor.zig");
pub const ops = @import("core/ops.zig");

// Re-export main types
pub const Tensor = tensor.Tensor;
pub const TensorError = tensor.TensorError;

pub fn main() !void {
    const print = std.debug.print;

    print("\n", .{});
    print("============================================================\n", .{});
    print("                    TOMOUL v0.1.0                           \n", .{});
    print("         Minimalist AI Inference Engine in Zig              \n", .{});
    print("============================================================\n", .{});
    print("\n", .{});

    // Use a general purpose allocator for the demo
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Demo: Create and manipulate tensors
    print("=== Phase 1: Tensor Core Demo ===\n\n", .{});

    // Create a 2x3 tensor
    print("1. Creating a 2x3 tensor...\n", .{});
    var shape = [_]usize{ 2, 3 };
    var a = try Tensor.init(allocator, &shape);
    defer a.deinit();

    // Fill with values
    for (a.data, 0..) |*val, i| {
        val.* = @floatFromInt(i + 1);
    }

    print("   Tensor A (2x3):\n", .{});
    print("   [ ", .{});
    for (a.data[0..3]) |v| print("{d:.1} ", .{v});
    print("]\n", .{});
    print("   [ ", .{});
    for (a.data[3..6]) |v| print("{d:.1} ", .{v});
    print("]\n\n", .{});

    // Create another tensor for operations
    print("2. Creating tensor B with same shape, filled with 10.0...\n", .{});
    var b = try Tensor.init(allocator, &shape);
    defer b.deinit();
    b.fill(10.0);

    print("   Tensor B (2x3):\n", .{});
    print("   [ ", .{});
    for (b.data[0..3]) |v| print("{d:.1} ", .{v});
    print("]\n", .{});
    print("   [ ", .{});
    for (b.data[3..6]) |v| print("{d:.1} ", .{v});
    print("]\n\n", .{});

    // Element-wise addition
    print("3. Element-wise addition: C = A + B\n", .{});
    var c = try ops.add(allocator, &a, &b);
    defer c.deinit();

    print("   Result C (2x3):\n", .{});
    print("   [ ", .{});
    for (c.data[0..3]) |v| print("{d:.1} ", .{v});
    print("]\n", .{});
    print("   [ ", .{});
    for (c.data[3..6]) |v| print("{d:.1} ", .{v});
    print("]\n\n", .{});

    // Element-wise multiplication
    print("4. Element-wise multiplication: D = A * B\n", .{});
    var d = try ops.mul(allocator, &a, &b);
    defer d.deinit();

    print("   Result D (2x3):\n", .{});
    print("   [ ", .{});
    for (d.data[0..3]) |v| print("{d:.1} ", .{v});
    print("]\n", .{});
    print("   [ ", .{});
    for (d.data[3..6]) |v| print("{d:.1} ", .{v});
    print("]\n\n", .{});

    // Scalar operations
    print("5. Scalar multiplication: E = A * 2.0\n", .{});
    var e = try ops.scale(allocator, &a, 2.0);
    defer e.deinit();

    print("   Result E (2x3):\n", .{});
    print("   [ ", .{});
    for (e.data[0..3]) |v| print("{d:.1} ", .{v});
    print("]\n", .{});
    print("   [ ", .{});
    for (e.data[3..6]) |v| print("{d:.1} ", .{v});
    print("]\n\n", .{});

    // Reduction operations
    print("6. Reduction operations on tensor A:\n", .{});
    print("   Sum:  {d:.1}\n", .{ops.sum(&a)});
    print("   Mean: {d:.2}\n", .{ops.mean(&a)});
    print("   Max:  {d:.1}\n", .{ops.max(&a)});
    print("   Min:  {d:.1}\n", .{ops.min(&a)});

    print("\n=== Phase 1 Complete! ===\n\n", .{});

    // =========================================================================
    // Phase 2: Matrix Multiplication Demo
    // =========================================================================
    print("=== Phase 2: Matrix Multiplication ===\n\n", .{});

    // Create Matrix X (1x3) -> [1, 2, 3]
    print("1. Matrix multiplication: Y = X @ W\n", .{});
    var shape_x = [_]usize{ 1, 3 };
    const data_x = [_]f32{ 1.0, 2.0, 3.0 };
    var mat_x = try Tensor.initWithData(allocator, &shape_x, &data_x);
    defer mat_x.deinit();

    print("   X (1x3): [ ", .{});
    for (mat_x.data) |v| print("{d:.1} ", .{v});
    print("]\n", .{});

    // Create Matrix W (3x2) -> [[1, 2], [3, 4], [5, 6]]
    var shape_w = [_]usize{ 3, 2 };
    const data_w = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var mat_w = try Tensor.initWithData(allocator, &shape_w, &data_w);
    defer mat_w.deinit();

    print("   W (3x2):\n", .{});
    print("   [ {d:.1} {d:.1} ]\n", .{ mat_w.data[0], mat_w.data[1] });
    print("   [ {d:.1} {d:.1} ]\n", .{ mat_w.data[2], mat_w.data[3] });
    print("   [ {d:.1} {d:.1} ]\n", .{ mat_w.data[4], mat_w.data[5] });

    // Perform Y = X @ W
    // Y[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
    // Y[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
    var mat_y = try ops.matmul(allocator, &mat_x, &mat_w);
    defer mat_y.deinit();

    print("\n   Y = X @ W (1x2):\n", .{});
    print("   [ {d:.1} {d:.1} ]  (expected: [22.0, 28.0])\n\n", .{ mat_y.data[0], mat_y.data[1] });

    // =========================================================================
    // Phase 2: Activation Functions Demo
    // =========================================================================
    print("=== Phase 2: Activation Functions ===\n\n", .{});

    // Create tensor [-1.0, 0.0, 1.0]
    var shape_act = [_]usize{3};
    const data_act = [_]f32{ -1.0, 0.0, 1.0 };
    var act_input = try Tensor.initWithData(allocator, &shape_act, &data_act);
    defer act_input.deinit();

    print("2. Input tensor: [ {d:.1} {d:.1} {d:.1} ]\n\n", .{ act_input.data[0], act_input.data[1], act_input.data[2] });

    // ReLU: max(0, x)
    var relu_out = try ops.relu(allocator, &act_input);
    defer relu_out.deinit();
    print("   ReLU(x):    [ {d:.3} {d:.3} {d:.3} ]  (expected: [0, 0, 1])\n", .{ relu_out.data[0], relu_out.data[1], relu_out.data[2] });

    // Sigmoid: 1 / (1 + exp(-x))
    var sig_out = try ops.sigmoid(allocator, &act_input);
    defer sig_out.deinit();
    print("   Sigmoid(x): [ {d:.3} {d:.3} {d:.3} ]  (expected: [0.269, 0.5, 0.731])\n", .{ sig_out.data[0], sig_out.data[1], sig_out.data[2] });

    // Tanh
    var tanh_out = try ops.tanh(allocator, &act_input);
    defer tanh_out.deinit();
    print("   Tanh(x):    [ {d:.3} {d:.3} {d:.3} ]  (expected: [-0.762, 0, 0.762])\n", .{ tanh_out.data[0], tanh_out.data[1], tanh_out.data[2] });

    print("\n=== Phase 2 Complete! ===\n", .{});
    print("Matrix multiplication and activations operational.\n", .{});
    print("Ready for Phase 3 (Data Bridge - PyTorch weight loading).\n\n", .{});
}

// Pull in tests from submodules
test {
    _ = @import("core/tensor.zig");
    _ = @import("core/ops.zig");
}
