// src/models/qwen3_5/deltanet.zig
// Gated DeltaNet implementation for Qwen3.5
//
// Replaces standard attention in 18 of 24 layers. Instead of KV cache,
// maintains a recurrent state matrix S[v_dim, k_dim] per head that is
// O(1) per token — doesn't grow with sequence length.
//
// Per-step recurrence:
//   1. Project: QKV = x @ W_qkv, z = x @ W_z, β = sigmoid(x @ W_b), α = x @ W_a
//   2. Conv1D on QKV (kernel=4, causal)
//   3. SiLU activation
//   4. Split → Q, K, V (16 heads each, 128 dim)
//   5. Gating: g = -exp(A_log) * softplus(α + dt_bias)
//   6. Per-head recurrence:
//      - L2 normalize Q, K
//      - S = exp(g) * S + β * (v ⊗ k) + (1-β) * (S·k) ⊗ k   (delta rule)
//      - output = S @ q
//   7. GroupNorm + output gate (z)
//   8. Output projection

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");
const config = @import("config.zig");
const Qwen3_5Config = config.Qwen3_5Config;
const model_types = @import("model.zig");
const ProjectionWeight = model_types.ProjectionWeight;
const projectMul = model_types.projectMul;

pub const DeltaNetWeights = struct {
    // Projections
    in_proj_qkv: ProjectionWeight, // [qkv_dim, hidden_size] = [6144, 1024]
    in_proj_z: ProjectionWeight, // [value_dim, hidden_size] = [2048, 1024]
    in_proj_b: ProjectionWeight, // [num_v_heads, hidden_size] = [16, 1024]
    in_proj_a: ProjectionWeight, // [num_v_heads, hidden_size] = [16, 1024]

    // Causal Conv1D (depthwise, kernel=4)
    conv1d_weight: Tensor, // [qkv_dim, kernel_size] = [6144, 4] (reshaped from [6144,1,4])
    conv1d_bias: Tensor, // [qkv_dim] = [6144]

    // Gating parameters
    A_log: Tensor, // [num_v_heads] = [16]
    dt_bias: Tensor, // [num_v_heads] = [16]

    // Output
    out_proj: ProjectionWeight, // [hidden_size, value_dim] = [1024, 2048]
    norm_weight: Tensor, // [value_dim] = [2048]
};

pub const DeltaNetState = struct {
    /// Recurrent state: [num_v_heads × v_head_dim × k_head_dim]
    /// For 0.8B: [16 × 128 × 128] = 256KB per layer
    recurrent: []f32,

    /// Conv state buffer: [qkv_dim × (kernel_size - 1)]
    /// For 0.8B: [6144 × 3] ≈ 73KB per layer
    conv: []f32,

    num_v_heads: usize,
    v_head_dim: usize,
    k_head_dim: usize,
    qkv_dim: usize,
    kernel_size: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, cfg: Qwen3_5Config) !DeltaNetState {
        const num_v_heads = cfg.linear_num_value_heads;
        const v_head_dim = cfg.linear_value_head_dim;
        const k_head_dim = cfg.linear_key_head_dim;
        const qkv_dim = cfg.linearQkvDim();
        const kernel_size = cfg.linear_conv_kernel_dim;

        const recurrent_size = num_v_heads * v_head_dim * k_head_dim;
        const conv_size = qkv_dim * (kernel_size - 1);

        const recurrent = try allocator.alloc(f32, recurrent_size);
        @memset(recurrent, 0);

        const conv = try allocator.alloc(f32, conv_size);
        @memset(conv, 0);

        return .{
            .recurrent = recurrent,
            .conv = conv,
            .num_v_heads = num_v_heads,
            .v_head_dim = v_head_dim,
            .k_head_dim = k_head_dim,
            .qkv_dim = qkv_dim,
            .kernel_size = kernel_size,
            .allocator = allocator,
        };
    }

    pub fn reset(self: *DeltaNetState) void {
        @memset(self.recurrent, 0);
        @memset(self.conv, 0);
    }

    pub fn deinit(self: *DeltaNetState) void {
        self.allocator.free(self.recurrent);
        self.allocator.free(self.conv);
    }
};

/// Pre-allocated scratch buffers for DeltaNet step, avoiding per-step allocation.
pub const DeltaNetScratch = struct {
    /// [qkv_dim] for mixed QKV projection + conv output
    mixed_qkv: []f32,
    /// [value_dim] for z gate
    z: []f32,
    /// [num_v_heads] for beta
    beta: []f32,
    /// [num_v_heads] for alpha
    alpha: []f32,
    /// [num_v_heads] for gating
    g: []f32,
    /// [value_dim] for DeltaNet output (before projection)
    output: []f32,
    /// [k_head_dim] for normalized Q
    q_norm: []f32,
    /// [k_head_dim] for normalized K
    k_norm: []f32,
    /// [v_head_dim] for S @ k_norm intermediate
    sk: []f32,
    /// [hidden_size] for final result
    result: []f32,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, cfg: Qwen3_5Config) !DeltaNetScratch {
        return .{
            .mixed_qkv = try allocator.alloc(f32, cfg.linearQkvDim()),
            .z = try allocator.alloc(f32, cfg.linearValueDim()),
            .beta = try allocator.alloc(f32, cfg.linear_num_value_heads),
            .alpha = try allocator.alloc(f32, cfg.linear_num_value_heads),
            .g = try allocator.alloc(f32, cfg.linear_num_value_heads),
            .output = try allocator.alloc(f32, cfg.linearValueDim()),
            .q_norm = try allocator.alloc(f32, cfg.linear_key_head_dim),
            .k_norm = try allocator.alloc(f32, cfg.linear_key_head_dim),
            .sk = try allocator.alloc(f32, cfg.linear_value_head_dim),
            .result = try allocator.alloc(f32, cfg.hidden_size),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DeltaNetScratch) void {
        self.allocator.free(self.mixed_qkv);
        self.allocator.free(self.z);
        self.allocator.free(self.beta);
        self.allocator.free(self.alpha);
        self.allocator.free(self.g);
        self.allocator.free(self.output);
        self.allocator.free(self.q_norm);
        self.allocator.free(self.k_norm);
        self.allocator.free(self.sk);
        self.allocator.free(self.result);
    }
};

/// Single-step DeltaNet forward pass.
/// hidden: [hidden_size] input for this token (must be RMSNorm'd by caller)
/// state: mutable recurrent + conv state for this layer
/// weights: this layer's DeltaNet weights
/// scratch: pre-allocated scratch buffers
/// do_out_proj: when true, write final out_proj into scratch.result; when false,
///              leave attention output in scratch.output (caller will run out_proj,
///              e.g. as part of a GPU-fused o_proj+FFN command buffer).
pub fn deltaNetStep(
    hidden: []const f32,
    state: *DeltaNetState,
    weights: *const DeltaNetWeights,
    scratch: *DeltaNetScratch,
    cfg: Qwen3_5Config,
    do_out_proj: bool,
) void {
    const hidden_size = cfg.hidden_size;
    const qkv_dim = cfg.linearQkvDim();
    const value_dim = cfg.linearValueDim();
    const num_v_heads = cfg.linear_num_value_heads;

    // ---- 1. Projections ----
    // mixed_qkv = W_qkv @ hidden  [qkv_dim]
    projectMul(scratch.mixed_qkv, weights.in_proj_qkv, hidden, qkv_dim, hidden_size);

    // z = W_z @ hidden  [value_dim]
    projectMul(scratch.z, weights.in_proj_z, hidden, value_dim, hidden_size);

    // beta = W_b @ hidden  [num_v_heads] (sigmoid applied in postProjections)
    projectMul(scratch.beta, weights.in_proj_b, hidden, num_v_heads, hidden_size);

    // alpha = W_a @ hidden  [num_v_heads]
    projectMul(scratch.alpha, weights.in_proj_a, hidden, num_v_heads, hidden_size);

    // Continue with conv1d, recurrence, output projection
    deltaNetStepPostProjections(state, weights, scratch, cfg, do_out_proj);
}

/// Continue DeltaNet step after projections are already computed.
/// Assumes scratch.mixed_qkv, z, beta, alpha are filled (beta is raw, not sigmoid'd).
/// Applies sigmoid to beta, then conv1d, SiLU, recurrence, norm, gate, out_proj.
/// do_out_proj: see deltaNetStep doc.
pub fn deltaNetStepPostProjections(
    state: *DeltaNetState,
    weights: *const DeltaNetWeights,
    scratch: *DeltaNetScratch,
    cfg: Qwen3_5Config,
    do_out_proj: bool,
) void {
    const hidden_size = cfg.hidden_size;
    const qkv_dim = cfg.linearQkvDim();
    const value_dim = cfg.linearValueDim();
    const key_dim = cfg.linearKeyDim();
    const num_v_heads = cfg.linear_num_value_heads;
    const v_head_dim = cfg.linear_value_head_dim;
    const k_head_dim = cfg.linear_key_head_dim;
    const kernel_size = cfg.linear_conv_kernel_dim;

    // Sigmoid on beta (raw linear projection → gate probability)
    for (scratch.beta) |*v| v.* = 1.0 / (1.0 + @exp(-v.*));

    // ---- 2. Causal Conv1D (single step with state) ----
    const ks_m1 = kernel_size - 1;
    for (0..qkv_dim) |ch| {
        const conv_row = state.conv[ch * ks_m1 ..][0..ks_m1];
        const w_base = ch * kernel_size;

        // Compute conv output FIRST using old state + current input
        var sum_val: f32 = weights.conv1d_bias.data[ch];
        for (0..ks_m1) |ji| {
            sum_val += conv_row[ji] * weights.conv1d_weight.data[w_base + ji];
        }
        sum_val += scratch.mixed_qkv[ch] * weights.conv1d_weight.data[w_base + ks_m1];

        // THEN update state: shift left, append current input
        var j: usize = 0;
        while (j < ks_m1 - 1) : (j += 1) {
            conv_row[j] = conv_row[j + 1];
        }
        conv_row[ks_m1 - 1] = scratch.mixed_qkv[ch];
        scratch.mixed_qkv[ch] = sum_val;
    }

    // SiLU activation on conv output
    ops.siluSliceInPlace(scratch.mixed_qkv);

    // ---- 3. Split into Q, K, V ----
    const q = scratch.mixed_qkv[0..key_dim];
    const k = scratch.mixed_qkv[key_dim..][0..key_dim];
    const v = scratch.mixed_qkv[key_dim + key_dim ..][0..value_dim];

    // ---- 4. Compute gating ----
    // g = -exp(A_log) * softplus(alpha + dt_bias)
    for (0..num_v_heads) |h| {
        const a_val = -@exp(weights.A_log.data[h]);
        scratch.g[h] = a_val * ops.softplus(scratch.alpha[h] + weights.dt_bias.data[h]);
    }

    // ---- 5. Recurrence per head ----
    for (0..num_v_heads) |h| {
        // Per-head slices
        const q_h = q[h * k_head_dim ..][0..k_head_dim];
        const k_h = k[h * k_head_dim ..][0..k_head_dim];
        const v_h = v[h * v_head_dim ..][0..v_head_dim];
        const o_h = scratch.output[h * v_head_dim ..][0..v_head_dim];

        // L2 normalize Q and K (into scratch buffers)
        @memcpy(scratch.q_norm, q_h);
        ops.l2NormInPlace(scratch.q_norm);
        // Scale Q by 1/sqrt(k_head_dim) per delta rule convention
        const q_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(k_head_dim)));
        for (scratch.q_norm) |*qv| qv.* *= q_scale;

        @memcpy(scratch.k_norm, k_h);
        ops.l2NormInPlace(scratch.k_norm);

        // State pointer for this head: S[v_head_dim, k_head_dim]
        const s = state.recurrent[h * v_head_dim * k_head_dim ..][0 .. v_head_dim * k_head_dim];

        // Decay existing state: S *= exp(g)
        const decay = @exp(scratch.g[h]);
        for (s) |*sv| sv.* *= decay;

        // Compute kv_mem = S @ k_norm (AFTER decay)
        ops.matvecMul(scratch.sk, s, scratch.k_norm, v_head_dim, k_head_dim);

        // Delta correction: delta = (v - kv_mem) * beta
        const b = scratch.beta[h];
        for (0..v_head_dim) |vi| {
            scratch.sk[vi] = (v_h[vi] - scratch.sk[vi]) * b;
        }

        // Update state: S += outer(delta, k_norm)
        ops.outerProductAddInPlace(s, scratch.sk, scratch.k_norm, v_head_dim, k_head_dim, 1.0);

        // Read from state: output = S @ q_norm (already scaled)
        ops.matvecMul(o_h, s, scratch.q_norm, v_head_dim, k_head_dim);
    }

    // ---- 6. Group normalization + output gate (z) ----
    ops.groupNormGatedInPlace(
        scratch.output,
        scratch.z,
        weights.norm_weight.data,
        value_dim,
        num_v_heads,
        cfg.rms_norm_eps,
    );

    // ---- 7. Output projection (optional — caller may skip to fuse on GPU) ----
    if (do_out_proj) {
        // result = W_out @ output  [hidden_size]
        projectMul(scratch.result, weights.out_proj, scratch.output, hidden_size, value_dim);
    }
}
