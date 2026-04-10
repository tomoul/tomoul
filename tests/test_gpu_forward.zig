// tests/test_gpu_forward.zig
//
// Test GPU forward pass against CPU reference implementation.
// Uses small synthetic weights (hidden=8, heads=2, ffn=16, 1 layer) for fast testing.

const std = @import("std");
const gpu_fwd = @import("vulkan_forward");
const gpu = @import("vulkan");

// Small model config for testing
const TEST_HIDDEN = 8;
const TEST_HEADS = 2;
const TEST_HEAD_DIM = 4;
const TEST_FFN = 16;
const TEST_LAYERS = 1;
const TEST_VOCAB = 32;
const TEST_MAX_SEQ = 16;

const test_config = gpu_fwd.GpuConfig{
    .hidden_dim = TEST_HIDDEN,
    .num_heads = TEST_HEADS,
    .head_dim = TEST_HEAD_DIM,
    .ffn_dim = TEST_FFN,
    .num_layers = TEST_LAYERS,
    .vocab_size = TEST_VOCAB,
    .max_seq_len = TEST_MAX_SEQ,
};

// CPU reference implementations for validation
fn cpuMatmulBias(a: []const f32, b: []const f32, bias: []const f32, out: []f32, m: usize, n: usize, k: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |kk| {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = sum + bias[j];
        }
    }
}

fn cpuLayerNorm(data: []f32, gamma: []const f32, beta: []const f32, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        const base = r * cols;
        var mean: f64 = 0;
        for (0..cols) |c| mean += data[base + c];
        mean /= @floatFromInt(cols);
        var variance: f64 = 0;
        for (0..cols) |c| {
            const d = @as(f64, data[base + c]) - mean;
            variance += d * d;
        }
        variance /= @floatFromInt(cols);
        const inv_std: f64 = 1.0 / @sqrt(variance + 1e-12);
        for (0..cols) |c| {
            data[base + c] = @floatCast((@as(f64, data[base + c]) - mean) * inv_std * gamma[c] + beta[c]);
        }
    }
}

fn cpuGelu(data: []f32) void {
    for (data) |*x| {
        const v = x.*;
        const x3 = v * v * v;
        const inner = 0.7978845608 * (v + 0.044715 * x3);
        const t = std.math.clamp(inner, -5.0, 5.0);
        const t2 = t * t;
        const tanh_val = t * (27.0 + t2) / (27.0 + 9.0 * t2);
        x.* = 0.5 * v * (1.0 + tanh_val);
    }
}

fn cpuAttention(q: []const f32, k: []const f32, v: []const f32, out: []f32, seq_len: usize, num_heads: usize, head_dim: usize) void {
    const hidden = num_heads * head_dim;
    for (0..num_heads) |h| {
        const ho = h * head_dim;
        for (0..seq_len) |i| {
            // Compute scores
            var scores: [TEST_MAX_SEQ]f32 = undefined;
            var max_score: f32 = -1e30;
            for (0..seq_len) |j| {
                var dp: f32 = 0;
                for (0..head_dim) |d| {
                    dp += q[i * hidden + ho + d] * k[j * hidden + ho + d];
                }
                const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
                scores[j] = dp * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }
            // Softmax
            var sum_exp: f32 = 0;
            for (0..seq_len) |j| {
                scores[j] = @exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            for (0..seq_len) |j| scores[j] /= sum_exp;
            // Weighted sum
            for (0..head_dim) |d| {
                var val: f32 = 0;
                for (0..seq_len) |j| {
                    val += scores[j] * v[j * hidden + ho + d];
                }
                out[i * hidden + ho + d] = val;
            }
        }
    }
}

fn cpuPoolNorm(data: []const f32, out: []f32, seq_len: usize, hidden: usize) void {
    // Mean pooling
    for (0..hidden) |d| {
        var sum: f32 = 0;
        for (0..seq_len) |t| sum += data[t * hidden + d];
        out[d] = sum / @as(f32, @floatFromInt(seq_len));
    }
    // L2 normalize
    var l2: f32 = 0;
    for (0..hidden) |d| l2 += out[d] * out[d];
    l2 = @sqrt(l2);
    if (l2 > 1e-12) {
        for (0..hidden) |d| out[d] /= l2;
    }
}

test "gpu layernorm correctness" {
    var ctx = gpu.VulkanContext.init(std.testing.allocator) catch return;
    defer ctx.deinit();

    const rows: u32 = 4;
    const cols: u32 = TEST_HIDDEN;
    const n = rows * cols;

    var data_cpu: [4 * TEST_HIDDEN]f32 = undefined;
    var gamma: [TEST_HIDDEN]f32 = undefined;
    var beta: [TEST_HIDDEN]f32 = undefined;

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (&data_cpu) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (&gamma) |*v| v.* = random.float(f32) * 0.5 + 0.75;
    for (&beta) |*v| v.* = random.float(f32) * 0.2 - 0.1;

    // GPU path
    var data_buf = try ctx.createStorageBuffer(n * 4, true);
    defer ctx.destroyBuffer(&data_buf);
    var gamma_buf = try ctx.createStorageBuffer(cols * 4, true);
    defer ctx.destroyBuffer(&gamma_buf);
    var beta_buf = try ctx.createStorageBuffer(cols * 4, true);
    defer ctx.destroyBuffer(&beta_buf);

    var data_gpu = data_cpu; // copy for GPU
    try ctx.uploadToBuffer(&data_buf, std.mem.sliceAsBytes(&data_gpu));
    try ctx.uploadToBuffer(&gamma_buf, std.mem.sliceAsBytes(&gamma));
    try ctx.uploadToBuffer(&beta_buf, std.mem.sliceAsBytes(&beta));

    const spirv = try loadShader("src/gpu/shaders/layernorm.spv");
    defer std.testing.allocator.free(spirv);
    var pipe = try ctx.createComputePipeline(spirv, 3, 12);
    defer ctx.destroyPipeline(&pipe);

    const desc = try ctx.allocateDescriptorSet(&pipe);
    try ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{ data_buf, gamma_buf, beta_buf });

    const PushC = extern struct { rows_: u32, cols_: u32, eps: f32 };
    const pc = PushC{ .rows_ = rows, .cols_ = cols, .eps = 1e-12 };
    try ctx.dispatch(&pipe, desc, rows, 1, 1, std.mem.asBytes(&pc));

    try ctx.readbackFromBuffer(&data_buf, std.mem.sliceAsBytes(&data_gpu));

    // CPU reference
    cpuLayerNorm(&data_cpu, &gamma, &beta, rows, cols);

    var max_err: f32 = 0;
    for (0..n) |i| {
        const err = @abs(data_gpu[i] - data_cpu[i]);
        if (err > max_err) max_err = err;
    }
    std.debug.print("layernorm: max_err={e}, ", .{max_err});
    try std.testing.expect(max_err < 1e-4);
    std.debug.print("PASS\n", .{});
}

test "gpu gelu correctness" {
    var ctx = gpu.VulkanContext.init(std.testing.allocator) catch return;
    defer ctx.deinit();

    const N: u32 = 256;
    var data_cpu: [N]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(123);
    const random = rng.random();
    for (&data_cpu) |*v| v.* = random.float(f32) * 6.0 - 3.0;

    var data_gpu = data_cpu;
    var buf = try ctx.createStorageBuffer(N * 4, true);
    defer ctx.destroyBuffer(&buf);
    try ctx.uploadToBuffer(&buf, std.mem.sliceAsBytes(&data_gpu));

    const spirv = try loadShader("src/gpu/shaders/gelu.spv");
    defer std.testing.allocator.free(spirv);
    var pipe = try ctx.createComputePipeline(spirv, 1, 4);
    defer ctx.destroyPipeline(&pipe);

    const desc = try ctx.allocateDescriptorSet(&pipe);
    try ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{buf});

    const pc: u32 = N;
    try ctx.dispatch(&pipe, desc, 1, 1, 1, std.mem.asBytes(&pc));

    try ctx.readbackFromBuffer(&buf, std.mem.sliceAsBytes(&data_gpu));
    cpuGelu(&data_cpu);

    var max_err: f32 = 0;
    for (0..N) |i| {
        const err = @abs(data_gpu[i] - data_cpu[i]);
        if (err > max_err) max_err = err;
    }
    std.debug.print("gelu: max_err={e}, ", .{max_err});
    try std.testing.expect(max_err < 1e-4);
    std.debug.print("PASS\n", .{});
}

test "gpu attention correctness" {
    var ctx = gpu.VulkanContext.init(std.testing.allocator) catch return;
    defer ctx.deinit();

    const seq: u32 = 4;
    const hidden: u32 = TEST_HIDDEN;
    const n = seq * hidden;

    var q_data: [4 * TEST_HIDDEN]f32 = undefined;
    var k_data: [4 * TEST_HIDDEN]f32 = undefined;
    var v_data: [4 * TEST_HIDDEN]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(77);
    const random = rng.random();
    for (&q_data) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (&k_data) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (&v_data) |*v| v.* = random.float(f32) * 2.0 - 1.0;

    var q_buf = try ctx.createStorageBuffer(n * 4, true);
    defer ctx.destroyBuffer(&q_buf);
    var k_buf = try ctx.createStorageBuffer(n * 4, true);
    defer ctx.destroyBuffer(&k_buf);
    var v_buf = try ctx.createStorageBuffer(n * 4, true);
    defer ctx.destroyBuffer(&v_buf);
    var o_buf = try ctx.createStorageBuffer(n * 4, true);
    defer ctx.destroyBuffer(&o_buf);

    try ctx.uploadToBuffer(&q_buf, std.mem.sliceAsBytes(&q_data));
    try ctx.uploadToBuffer(&k_buf, std.mem.sliceAsBytes(&k_data));
    try ctx.uploadToBuffer(&v_buf, std.mem.sliceAsBytes(&v_data));

    const spirv = try loadShader("src/gpu/shaders/attention.spv");
    defer std.testing.allocator.free(spirv);
    var pipe = try ctx.createComputePipeline(spirv, 4, 16);
    defer ctx.destroyPipeline(&pipe);

    const desc = try ctx.allocateDescriptorSet(&pipe);
    try ctx.bindBuffers(desc, &[_]gpu.GpuBuffer{ q_buf, k_buf, v_buf, o_buf });

    const AttnPC = extern struct { sl: u32, nh: u32, hd: u32, scale: f32 };
    const pc = AttnPC{ .sl = seq, .nh = TEST_HEADS, .hd = TEST_HEAD_DIM, .scale = 1.0 / @sqrt(@as(f32, TEST_HEAD_DIM)) };
    try ctx.dispatch(&pipe, desc, TEST_HEADS, seq, 1, std.mem.asBytes(&pc));

    var gpu_out: [4 * TEST_HIDDEN]f32 = undefined;
    try ctx.readbackFromBuffer(&o_buf, std.mem.sliceAsBytes(&gpu_out));

    var cpu_out: [4 * TEST_HIDDEN]f32 = undefined;
    cpuAttention(&q_data, &k_data, &v_data, &cpu_out, seq, TEST_HEADS, TEST_HEAD_DIM);

    var max_err: f32 = 0;
    for (0..n) |i| {
        const err = @abs(gpu_out[i] - cpu_out[i]);
        if (err > max_err) max_err = err;
    }
    std.debug.print("attention: max_err={e}, ", .{max_err});
    try std.testing.expect(max_err < 1e-4);
    std.debug.print("PASS\n", .{});
}

test "gpu forward pass end-to-end" {
    const allocator = std.testing.allocator;

    // Generate synthetic weights
    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();

    // Embedding weights
    var word_emb: [TEST_VOCAB * TEST_HIDDEN]f32 = undefined;
    var pos_emb: [TEST_MAX_SEQ * TEST_HIDDEN]f32 = undefined;
    var type_emb: [2 * TEST_HIDDEN]f32 = undefined;
    var embed_gamma: [TEST_HIDDEN]f32 = undefined;
    var embed_beta: [TEST_HIDDEN]f32 = undefined;
    for (&word_emb) |*v| v.* = random.float(f32) * 0.2 - 0.1;
    for (&pos_emb) |*v| v.* = random.float(f32) * 0.2 - 0.1;
    for (&type_emb) |*v| v.* = random.float(f32) * 0.2 - 0.1;
    for (&embed_gamma) |*v| v.* = random.float(f32) * 0.5 + 0.75;
    for (&embed_beta) |*v| v.* = random.float(f32) * 0.2 - 0.1;

    // Layer weights
    var q_w: [TEST_HIDDEN * TEST_HIDDEN]f32 = undefined;
    var q_b: [TEST_HIDDEN]f32 = undefined;
    var k_w: [TEST_HIDDEN * TEST_HIDDEN]f32 = undefined;
    var k_b: [TEST_HIDDEN]f32 = undefined;
    var v_w: [TEST_HIDDEN * TEST_HIDDEN]f32 = undefined;
    var v_b: [TEST_HIDDEN]f32 = undefined;
    var o_w: [TEST_HIDDEN * TEST_HIDDEN]f32 = undefined;
    var o_b: [TEST_HIDDEN]f32 = undefined;
    var ff1_w: [TEST_HIDDEN * TEST_FFN]f32 = undefined;
    var ff1_b: [TEST_FFN]f32 = undefined;
    var ff2_w: [TEST_FFN * TEST_HIDDEN]f32 = undefined;
    var ff2_b: [TEST_HIDDEN]f32 = undefined;
    var aln_g: [TEST_HIDDEN]f32 = undefined;
    var aln_b: [TEST_HIDDEN]f32 = undefined;
    var fln_g: [TEST_HIDDEN]f32 = undefined;
    var fln_b: [TEST_HIDDEN]f32 = undefined;

    inline for (.{
        &q_w, &k_w, &v_w, &o_w,
    }) |w| for (w) |*val| {
        val.* = random.float(f32) * 0.2 - 0.1;
    };
    inline for (.{
        &q_b, &k_b, &v_b, &o_b, &ff2_b,
    }) |b| for (b) |*val| {
        val.* = random.float(f32) * 0.1 - 0.05;
    };
    for (&ff1_w) |*v| v.* = random.float(f32) * 0.2 - 0.1;
    for (&ff1_b) |*v| v.* = random.float(f32) * 0.1 - 0.05;
    for (&ff2_w) |*v| v.* = random.float(f32) * 0.2 - 0.1;
    for (&aln_g) |*v| v.* = random.float(f32) * 0.5 + 0.75;
    for (&aln_b) |*v| v.* = random.float(f32) * 0.2 - 0.1;
    for (&fln_g) |*v| v.* = random.float(f32) * 0.5 + 0.75;
    for (&fln_b) |*v| v.* = random.float(f32) * 0.2 - 0.1;

    const layer = gpu_fwd.LayerData{
        .q_weight = &q_w, .q_bias = &q_b,
        .k_weight = &k_w, .k_bias = &k_b,
        .v_weight = &v_w, .v_bias = &v_b,
        .o_weight = &o_w, .o_bias = &o_b,
        .ff1_weight = &ff1_w, .ff1_bias = &ff1_b,
        .ff2_weight = &ff2_w, .ff2_bias = &ff2_b,
        .attn_ln_gamma = &aln_g, .attn_ln_beta = &aln_b,
        .ff_ln_gamma = &fln_g, .ff_ln_beta = &fln_b,
    };

    const embeddings = gpu_fwd.EmbeddingData{
        .word_emb = &word_emb,
        .pos_emb = &pos_emb,
        .type_emb = &type_emb,
        .ln_gamma = &embed_gamma,
        .ln_beta = &embed_beta,
    };

    // Init GPU forward
    var fwd = gpu_fwd.GpuForward.init(allocator, test_config, embeddings, &[_]gpu_fwd.LayerData{layer}) catch |e| {
        std.debug.print("GPU forward init failed (expected if no Vulkan): {}\n", .{e});
        return;
    };
    defer fwd.deinit();

    // Run GPU forward
    const token_ids = [_]u32{ 5, 12, 3, 8, 1 };
    const seq_len: usize = token_ids.len;
    var gpu_output: [TEST_HIDDEN]f32 = undefined;
    try fwd.forward(&token_ids, &gpu_output);

    // CPU reference forward pass
    var embedded: [5 * TEST_HIDDEN]f32 = undefined;
    for (0..seq_len) |t| {
        for (0..TEST_HIDDEN) |d| {
            embedded[t * TEST_HIDDEN + d] = word_emb[token_ids[t] * TEST_HIDDEN + d] +
                pos_emb[t * TEST_HIDDEN + d] +
                type_emb[0 * TEST_HIDDEN + d]; // type_id = 0
        }
    }
    cpuLayerNorm(&embedded, &embed_gamma, &embed_beta, seq_len, TEST_HIDDEN);

    // Transformer layer
    var q_out: [5 * TEST_HIDDEN]f32 = undefined;
    var k_out: [5 * TEST_HIDDEN]f32 = undefined;
    var v_out: [5 * TEST_HIDDEN]f32 = undefined;
    cpuMatmulBias(&embedded, &q_w, &q_b, &q_out, seq_len, TEST_HIDDEN, TEST_HIDDEN);
    cpuMatmulBias(&embedded, &k_w, &k_b, &k_out, seq_len, TEST_HIDDEN, TEST_HIDDEN);
    cpuMatmulBias(&embedded, &v_w, &v_b, &v_out, seq_len, TEST_HIDDEN, TEST_HIDDEN);

    var attn_out: [5 * TEST_HIDDEN]f32 = undefined;
    cpuAttention(&q_out, &k_out, &v_out, &attn_out, seq_len, TEST_HEADS, TEST_HEAD_DIM);

    var proj_out: [5 * TEST_HIDDEN]f32 = undefined;
    cpuMatmulBias(&attn_out, &o_w, &o_b, &proj_out, seq_len, TEST_HIDDEN, TEST_HIDDEN);

    // Residual + LayerNorm
    for (0..seq_len * TEST_HIDDEN) |i| proj_out[i] += embedded[i];
    cpuLayerNorm(&proj_out, &aln_g, &aln_b, seq_len, TEST_HIDDEN);

    // FFN
    var ff1_out: [5 * TEST_FFN]f32 = undefined;
    cpuMatmulBias(&proj_out, &ff1_w, &ff1_b, &ff1_out, seq_len, TEST_FFN, TEST_HIDDEN);
    cpuGelu(&ff1_out);

    var ff2_out: [5 * TEST_HIDDEN]f32 = undefined;
    cpuMatmulBias(&ff1_out, &ff2_w, &ff2_b, &ff2_out, seq_len, TEST_HIDDEN, TEST_FFN);

    // Residual + LayerNorm
    for (0..seq_len * TEST_HIDDEN) |i| ff2_out[i] += proj_out[i];
    cpuLayerNorm(&ff2_out, &fln_g, &fln_b, seq_len, TEST_HIDDEN);

    // Pool + Normalize
    var cpu_output: [TEST_HIDDEN]f32 = undefined;
    cpuPoolNorm(&ff2_out, &cpu_output, seq_len, TEST_HIDDEN);

    // Compare GPU vs CPU
    var max_err: f32 = 0;
    var avg_err: f64 = 0;
    for (0..TEST_HIDDEN) |i| {
        const err = @abs(gpu_output[i] - cpu_output[i]);
        if (err > max_err) max_err = err;
        avg_err += err;
    }
    avg_err /= TEST_HIDDEN;

    std.debug.print("forward pass: max_err={e}, avg_err={e}\n", .{ max_err, avg_err });
    std.debug.print("  GPU: ", .{});
    for (gpu_output[0..@min(TEST_HIDDEN, 4)]) |v| std.debug.print("{d:.4} ", .{v});
    std.debug.print("\n  CPU: ", .{});
    for (cpu_output[0..@min(TEST_HIDDEN, 4)]) |v| std.debug.print("{d:.4} ", .{v});
    std.debug.print("\n", .{});

    try std.testing.expect(max_err < 0.01);
    std.debug.print("PASS\n", .{});
}

fn loadShader(path: []const u8) ![]const u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    const data = try std.testing.allocator.alloc(u8, stat.size);
    const bytes_read = try file.readAll(data);
    return data[0..bytes_read];
}
