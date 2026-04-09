// tests/test_vulkan_compute.zig
// Smoke test for Vulkan compute pipeline
//
// Tests: device init → buffer create → shader load → dispatch → readback
// Uses the vec_add shader: C[i] = A[i] + B[i]
// Then tests SGEMM shader for a small matrix multiply.

const std = @import("std");
const gpu = @import("vulkan");

fn loadShader(path: []const u8) ![]const u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    const data = try std.testing.allocator.alloc(u8, stat.size);
    const bytes_read = try file.readAll(data);
    return data[0..bytes_read];
}

test "vulkan device initialization" {
    var ctx = gpu.VulkanContext.init(std.testing.allocator) catch |err| {
        std.debug.print("Vulkan init failed (expected if no GPU): {}\n", .{err});
        return; // Skip test if no Vulkan device
    };
    defer ctx.deinit();

    std.debug.print("Vulkan device: {s}\n", .{ctx.getDeviceName()});
    std.debug.print("Max workgroup size: {}x{}x{}\n", .{
        ctx.max_compute_work_group_size[0],
        ctx.max_compute_work_group_size[1],
        ctx.max_compute_work_group_size[2],
    });
    std.debug.print("Shared memory: {} bytes\n", .{ctx.max_compute_shared_memory});
    std.debug.print("Subgroup size: {}\n", .{ctx.subgroup_size});
}

test "vulkan vec_add smoke test" {
    var ctx = gpu.VulkanContext.init(std.testing.allocator) catch return;
    defer ctx.deinit();

    const vec_add_spirv = loadShader("src/gpu/shaders/vec_add.spv") catch {
        std.debug.print("SKIP: vec_add.spv not found (run glslangValidator first)\n", .{});
        return;
    };
    defer std.testing.allocator.free(vec_add_spirv);

    const N: u32 = 1024;
    const buf_size = N * @sizeOf(f32);

    // Create host-visible buffers for input/output
    const usage = @as(u32, 0x80); // VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    var buf_a = try ctx.createBuffer(buf_size, usage, true);
    defer ctx.destroyBuffer(&buf_a);
    var buf_b = try ctx.createBuffer(buf_size, usage, true);
    defer ctx.destroyBuffer(&buf_b);
    var buf_c = try ctx.createBuffer(buf_size, usage, true);
    defer ctx.destroyBuffer(&buf_c);

    // Prepare input data
    var data_a: [N]f32 = undefined;
    var data_b: [N]f32 = undefined;
    for (0..N) |i| {
        data_a[i] = @floatFromInt(i);
        data_b[i] = @as(f32, @floatFromInt(i)) * 2.0;
    }

    try ctx.uploadToBuffer(&buf_a, std.mem.sliceAsBytes(&data_a));
    try ctx.uploadToBuffer(&buf_b, std.mem.sliceAsBytes(&data_b));

    // Create pipeline
    var pipeline = try ctx.createComputePipeline(vec_add_spirv, 3, @sizeOf(u32));
    defer ctx.destroyPipeline(&pipeline);

    // Allocate and bind descriptor set
    const desc_set = try ctx.allocateDescriptorSet(&pipeline);
    try ctx.bindBuffers(desc_set, &[_]gpu.GpuBuffer{ buf_a, buf_b, buf_c });

    // Dispatch: 1024 elements / 256 threads per workgroup = 4 workgroups
    const push_data = std.mem.asBytes(&N);
    try ctx.dispatch(&pipeline, desc_set, (N + 255) / 256, 1, 1, push_data);

    // Readback result
    var result: [N]f32 = undefined;
    try ctx.readbackFromBuffer(&buf_c, std.mem.sliceAsBytes(&result));

    // Verify: C[i] = A[i] + B[i] = i + 2*i = 3*i
    var max_err: f32 = 0;
    for (0..N) |i| {
        const expected: f32 = @as(f32, @floatFromInt(i)) * 3.0;
        const err = @abs(result[i] - expected);
        if (err > max_err) max_err = err;
    }

    std.debug.print("vec_add: max error = {e}, ", .{max_err});
    try std.testing.expect(max_err < 1e-6);
    std.debug.print("PASS\n", .{});
}

test "vulkan sgemm smoke test" {
    var ctx = gpu.VulkanContext.init(std.testing.allocator) catch return;
    defer ctx.deinit();

    const sgemm_spirv = loadShader("src/gpu/shaders/sgemm.spv") catch {
        std.debug.print("SKIP: sgemm.spv not found\n", .{});
        return;
    };
    defer std.testing.allocator.free(sgemm_spirv);

    // Small SGEMM: C = 1.0 * A @ B + 0.0 * C
    // A: 4x8, B: 8x4, C: 4x4
    const M: u32 = 4;
    const N: u32 = 4;
    const K: u32 = 8;

    const size_a = M * K * @sizeOf(f32);
    const size_b = K * N * @sizeOf(f32);
    const size_c = M * N * @sizeOf(f32);

    const usage = @as(u32, 0x80); // VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    var buf_a = try ctx.createBuffer(size_a, usage, true);
    defer ctx.destroyBuffer(&buf_a);
    var buf_b = try ctx.createBuffer(size_b, usage, true);
    defer ctx.destroyBuffer(&buf_b);
    var buf_c = try ctx.createBuffer(size_c, usage, true);
    defer ctx.destroyBuffer(&buf_c);

    // A = identity-like pattern, B = simple values
    var data_a: [M * K]f32 = undefined;
    var data_b: [K * N]f32 = undefined;
    for (0..M * K) |i| {
        data_a[i] = if (i / K == i % K) 1.0 else 0.0; // identity (padded)
    }
    for (0..K * N) |i| {
        data_b[i] = @as(f32, @floatFromInt(i)) + 1.0;
    }

    try ctx.uploadToBuffer(&buf_a, std.mem.sliceAsBytes(&data_a));
    try ctx.uploadToBuffer(&buf_b, std.mem.sliceAsBytes(&data_b));

    // Push constants: M, N, K, alpha, beta
    const PushConstants = extern struct {
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        beta: f32,
    };
    const pc = PushConstants{ .m = M, .n = N, .k = K, .alpha = 1.0, .beta = 0.0 };

    // Create pipeline (push constants = 5 * 4 = 20 bytes)
    var pipeline = try ctx.createComputePipeline(sgemm_spirv, 3, @sizeOf(PushConstants));
    defer ctx.destroyPipeline(&pipeline);

    const desc_set = try ctx.allocateDescriptorSet(&pipeline);
    try ctx.bindBuffers(desc_set, &[_]gpu.GpuBuffer{ buf_a, buf_b, buf_c });

    // Dispatch: tile size 64x64, so just 1 workgroup for this tiny matrix
    try ctx.dispatch(&pipeline, desc_set, 1, 1, 1, std.mem.asBytes(&pc));

    // Readback
    var result: [M * N]f32 = undefined;
    try ctx.readbackFromBuffer(&buf_c, std.mem.sliceAsBytes(&result));

    // Compute expected: C = A @ B (A is identity padded, so C = first M rows of B)
    var expected: [M * N]f32 = undefined;
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |kk| {
                sum += data_a[i * K + kk] * data_b[kk * N + j];
            }
            expected[i * N + j] = sum;
        }
    }

    var max_err: f32 = 0;
    for (0..M * N) |i| {
        const err = @abs(result[i] - expected[i]);
        if (err > max_err) max_err = err;
    }

    std.debug.print("sgemm 4x4x8: max error = {e}, ", .{max_err});
    try std.testing.expect(max_err < 1e-4);
    std.debug.print("PASS\n", .{});
}

test "vulkan sgemm transformer shapes" {
    var ctx = gpu.VulkanContext.init(std.testing.allocator) catch return;
    defer ctx.deinit();

    const sgemm_spirv = loadShader("src/gpu/shaders/sgemm.spv") catch {
        std.debug.print("SKIP: sgemm.spv not found\n", .{});
        return;
    };
    defer std.testing.allocator.free(sgemm_spirv);

    // Test with typical transformer shape: [11, 384] @ [384, 384]
    const M: u32 = 11;
    const N: u32 = 384;
    const K: u32 = 384;

    const size_a = M * K * @sizeOf(f32);
    const size_b = K * N * @sizeOf(f32);
    const size_c = M * N * @sizeOf(f32);

    const usage = @as(u32, 0x80);
    var buf_a = try ctx.createBuffer(size_a, usage, true);
    defer ctx.destroyBuffer(&buf_a);
    var buf_b = try ctx.createBuffer(size_b, usage, true);
    defer ctx.destroyBuffer(&buf_b);
    var buf_c = try ctx.createBuffer(size_c, usage, true);
    defer ctx.destroyBuffer(&buf_c);

    // Fill with pseudo-random data
    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();

    const a_data = try std.testing.allocator.alloc(f32, M * K);
    defer std.testing.allocator.free(a_data);
    const b_data = try std.testing.allocator.alloc(f32, K * N);
    defer std.testing.allocator.free(b_data);

    for (a_data) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (b_data) |*v| v.* = random.float(f32) * 2.0 - 1.0;

    try ctx.uploadToBuffer(&buf_a, std.mem.sliceAsBytes(a_data));
    try ctx.uploadToBuffer(&buf_b, std.mem.sliceAsBytes(b_data));

    const PushConstants = extern struct {
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        beta: f32,
    };
    const pc = PushConstants{ .m = M, .n = N, .k = K, .alpha = 1.0, .beta = 0.0 };

    var pipeline = try ctx.createComputePipeline(sgemm_spirv, 3, @sizeOf(PushConstants));
    defer ctx.destroyPipeline(&pipeline);

    const desc_set = try ctx.allocateDescriptorSet(&pipeline);
    try ctx.bindBuffers(desc_set, &[_]gpu.GpuBuffer{ buf_a, buf_b, buf_c });

    // Dispatch: ceil(N/64) x ceil(M/64) workgroups
    const wg_x = (N + 63) / 64;
    const wg_y = (M + 63) / 64;
    try ctx.dispatch(&pipeline, desc_set, wg_x, wg_y, 1, std.mem.asBytes(&pc));

    // Readback
    const result = try std.testing.allocator.alloc(f32, M * N);
    defer std.testing.allocator.free(result);
    try ctx.readbackFromBuffer(&buf_c, std.mem.sliceAsBytes(result));

    // CPU reference
    const expected = try std.testing.allocator.alloc(f32, M * N);
    defer std.testing.allocator.free(expected);

    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |kk| {
                sum += a_data[i * K + kk] * b_data[kk * N + j];
            }
            expected[i * N + j] = sum;
        }
    }

    var max_err: f32 = 0;
    var avg_err: f64 = 0;
    for (0..M * N) |i| {
        const err = @abs(result[i] - expected[i]);
        if (err > max_err) max_err = err;
        avg_err += err;
    }
    avg_err /= @floatFromInt(M * N);

    std.debug.print("sgemm [11x384]@[384x384]: max_err={e}, avg_err={e}, ", .{ max_err, avg_err });
    // Allow slightly larger tolerance for GPU float differences
    try std.testing.expect(max_err < 0.01);
    std.debug.print("PASS\n", .{});
}
