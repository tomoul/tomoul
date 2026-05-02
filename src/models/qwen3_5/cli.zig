// src/models/qwen3_5/cli.zig
// CLI for Qwen3.5-0.8B text generation
//
// Usage:
//   zig build -Dmodel=qwen3_5-0.8b run -- generate "What is 2+2?" --weights model.tl --tokenizer vocab.bin

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");
const loader_mod = @import("loader.zig");
const ModelLoader = loader_mod.ModelLoader;

const model_mod = @import("model.zig");
const Qwen3_5 = model_mod.Qwen3_5;
const Qwen3_5Config = model_mod.config.Qwen3_5Config;
const SpecialTokens = model_mod.config.SpecialTokens;
const Tokenizer = model_mod.tokenizer_mod.Tokenizer;
const ProjectionWeight = model_mod.ProjectionWeight;
const qwen3_5_gpu = @import("qwen3_5_gpu");
const GpuAccelerator = qwen3_5_gpu.GpuAccelerator;

const VERSION = "0.1.0";
const DEFAULT_WEIGHTS_PATH = "artifacts/qwen3_5_0.8b_q8k.tl";
const DEFAULT_TOKENIZER_PATH = "artifacts/qwen3_5_vocab.bin";

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Initialize parallel context
    var ctx = ops.Context.initMultiThreaded(allocator, null);
    defer ctx.deinit();
    ops.initGlobalContext(&ctx);
    defer ops.deinitGlobalContext();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printUsage();
        return;
    }

    const command = args[1];

    // Parse options
    var weights_path: []const u8 = DEFAULT_WEIGHTS_PATH;
    var tokenizer_path: []const u8 = DEFAULT_TOKENIZER_PATH;
    var max_tokens: usize = 256;
    var max_cache_len: usize = 4096;
    var prompt: ?[]const u8 = null;
    var use_gpu: bool = false;

    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--weights") or std.mem.eql(u8, args[i], "-w")) {
            if (i + 1 < args.len) {
                weights_path = args[i + 1];
                i += 1;
            }
        } else if (std.mem.eql(u8, args[i], "--tokenizer") or std.mem.eql(u8, args[i], "-t")) {
            if (i + 1 < args.len) {
                tokenizer_path = args[i + 1];
                i += 1;
            }
        } else if (std.mem.eql(u8, args[i], "--max-tokens") or std.mem.eql(u8, args[i], "-n")) {
            if (i + 1 < args.len) {
                max_tokens = std.fmt.parseInt(usize, args[i + 1], 10) catch 256;
                i += 1;
            }
        } else if (std.mem.eql(u8, args[i], "--max-cache-len")) {
            if (i + 1 < args.len) {
                max_cache_len = std.fmt.parseInt(usize, args[i + 1], 10) catch 4096;
                i += 1;
            }
        } else if (std.mem.eql(u8, args[i], "--gpu")) {
            use_gpu = true;
        } else if (prompt == null) {
            prompt = args[i];
        }
    }

    if (std.mem.eql(u8, command, "generate")) {
        try runGenerate(allocator, weights_path, tokenizer_path, prompt orelse "Hello", max_tokens, max_cache_len, use_gpu);
    } else if (std.mem.eql(u8, command, "version")) {
        std.debug.print("Tomoul Qwen3.5-0.8B v{s}\n", .{VERSION});
    } else {
        printUsage();
    }
}

fn registerWeight(accel: *GpuAccelerator, w: ProjectionWeight) !void {
    switch (w) {
        .f32 => |t| try accel.registerF32(@intFromPtr(t.data.ptr), t.data, t.shape[0], t.shape[1]),
        .q8k => |q| try accel.registerQ8K(@intFromPtr(q.data.ptr), q.data, q.scales, q.shape[0], q.shape[1], q.block_size),
    }
}

fn weightKey(w: ProjectionWeight) usize {
    return switch (w) {
        .f32 => |t| @intFromPtr(t.data.ptr),
        .q8k => |q| @intFromPtr(q.data.ptr),
    };
}

fn runGenerate(
    allocator: std.mem.Allocator,
    weights_path: []const u8,
    tokenizer_path: []const u8,
    prompt: []const u8,
    max_tokens: usize,
    max_cache_len: usize,
    use_gpu: bool,
) !void {
    std.debug.print("Loading tokenizer from {s}...\n", .{tokenizer_path});
    var tokenizer = try Tokenizer.init(allocator, tokenizer_path);
    defer tokenizer.deinit();

    std.debug.print("Loading model from {s}...\n", .{weights_path});
    var loader = try ModelLoader.init(allocator, weights_path);
    defer loader.deinit();

    var model = try Qwen3_5.loadWithConfig(allocator, &loader, Qwen3_5Config.default, max_cache_len);
    defer model.deinit();

    // GPU acceleration (optional)
    var gpu_accel: GpuAccelerator = undefined;
    var gpu_active = false;
    defer if (gpu_active) {
        model_mod.setGpuMatvec(null);
        model_mod.setGpuFfn(null);
        model_mod.setGpuDnInput(null);
        model_mod.setGpuAttnInput(null);
        model_mod.setGpuOprojFfn(null);
        model_mod.setGpuTokenBegin(null);
        model_mod.setGpuTokenEnd(null);
        model_mod.setGpuLayerStartDn(null);
        model_mod.setGpuLayerStartFa(null);
        model_mod.setGpuOprojFfnResident(null);
        qwen3_5_gpu.setGlobalInstance(null);
        gpu_accel.deinit();
    };

    if (use_gpu) {
        std.debug.print("Initializing GPU...\n", .{});
        gpu_accel = try GpuAccelerator.init(allocator);
        gpu_active = true;

        // Register all projection weights for GPU dispatch
        try registerWeight(&gpu_accel, model.weights.embed_tokens);
        for (model.weights.layers) |*lw| {
            try registerWeight(&gpu_accel, lw.gate_proj);
            try registerWeight(&gpu_accel, lw.up_proj);
            try registerWeight(&gpu_accel, lw.down_proj);
            switch (lw.layer_type) {
                .linear_attention => {
                    const dn = lw.deltanet_w.?;
                    try registerWeight(&gpu_accel, dn.in_proj_qkv);
                    try registerWeight(&gpu_accel, dn.in_proj_z);
                    try registerWeight(&gpu_accel, dn.in_proj_b);
                    try registerWeight(&gpu_accel, dn.in_proj_a);
                    try registerWeight(&gpu_accel, dn.out_proj);
                },
                .full_attention => {
                    const fa = lw.full_attn.?;
                    try registerWeight(&gpu_accel, fa.q_proj);
                    try registerWeight(&gpu_accel, fa.k_proj);
                    try registerWeight(&gpu_accel, fa.v_proj);
                    try registerWeight(&gpu_accel, fa.o_proj);
                },
            }
        }

        gpu_accel.printStats();

        // Initialize fused FFN pipeline (RMSNorm + gate/up/SiLU/down + residual in one dispatch)
        const cfg = Qwen3_5Config.default;
        const ffn_layers = try allocator.alloc(qwen3_5_gpu.FfnLayerInfo, cfg.num_hidden_layers);
        defer allocator.free(ffn_layers);
        for (model.weights.layers, 0..) |*lw, i| {
            const o_proj_w = switch (lw.layer_type) {
                .linear_attention => lw.deltanet_w.?.out_proj,
                .full_attention => lw.full_attn.?.o_proj,
            };
            ffn_layers[i] = .{
                .norm_weight_data = lw.post_attn_layernorm.data,
                .gate_weight_key = weightKey(lw.gate_proj),
                .up_weight_key = weightKey(lw.up_proj),
                .down_weight_key = weightKey(lw.down_proj),
                .o_proj_weight_key = weightKey(o_proj_w),
            };
        }
        // Sized for largest attention output across all layers (q_dim or value_dim)
        const attn_out_capacity: u32 = @intCast(@max(
            cfg.fullAttentionQDim(),
            cfg.linearValueDim(),
        ));
        try gpu_accel.initFusedFfn(
            ffn_layers,
            @intCast(cfg.hidden_size),
            @intCast(cfg.intermediate_size),
            cfg.rms_norm_eps,
            attn_out_capacity,
        );

        // Initialize fused input projections (RMSNorm + attention/DeltaNet projections in one dispatch)
        const input_proj_layers = try allocator.alloc(qwen3_5_gpu.InputProjLayerInfo, cfg.num_hidden_layers);
        defer allocator.free(input_proj_layers);
        for (model.weights.layers, 0..) |*lw, li| {
            var info = qwen3_5_gpu.InputProjLayerInfo{
                .is_deltanet = (lw.layer_type == .linear_attention),
                .input_norm_data = lw.input_layernorm.data,
                .dn_qkv_key = 0,
                .dn_z_key = 0,
                .dn_b_key = 0,
                .dn_a_key = 0,
                .fa_q_key = 0,
                .fa_k_key = 0,
                .fa_v_key = 0,
            };
            switch (lw.layer_type) {
                .linear_attention => {
                    const dn = lw.deltanet_w.?;
                    info.dn_qkv_key = weightKey(dn.in_proj_qkv);
                    info.dn_z_key = weightKey(dn.in_proj_z);
                    info.dn_b_key = weightKey(dn.in_proj_b);
                    info.dn_a_key = weightKey(dn.in_proj_a);
                },
                .full_attention => {
                    const fa = lw.full_attn.?;
                    info.fa_q_key = weightKey(fa.q_proj);
                    info.fa_k_key = weightKey(fa.k_proj);
                    info.fa_v_key = weightKey(fa.v_proj);
                },
            }
            input_proj_layers[li] = info;
        }
        try gpu_accel.initFusedInputProj(
            input_proj_layers,
            @intCast(cfg.hidden_size),
            cfg.rms_norm_eps,
            @intCast(cfg.linearQkvDim()),
            @intCast(cfg.linearValueDim()),
            @intCast(cfg.linear_num_value_heads),
            @intCast(cfg.fullAttentionQProjDim()),
            @intCast(cfg.fullAttentionKvDim()),
        );

        // Phase 4: fused final RMSNorm + LM head (single submit at token end).
        try gpu_accel.initFusedLmHead(
            model.weights.final_norm.data,
            weightKey(model.weights.embed_tokens),
        );

        qwen3_5_gpu.setGlobalInstance(&gpu_accel);
        model_mod.setGpuMatvec(&GpuAccelerator.dispatch);
        model_mod.setGpuFfn(&GpuAccelerator.dispatchFfn);
        model_mod.setGpuDnInput(&GpuAccelerator.dispatchDnInput);
        model_mod.setGpuAttnInput(&GpuAccelerator.dispatchAttnInput);
        model_mod.setGpuOprojFfn(&GpuAccelerator.dispatchOprojFfn);
        // Phase 2B: GPU-resident hidden state (no per-layer readback/upload).
        model_mod.setGpuTokenBegin(&GpuAccelerator.dispatchTokenBegin);
        model_mod.setGpuTokenEnd(&GpuAccelerator.dispatchTokenEnd);
        model_mod.setGpuLayerStartDn(&GpuAccelerator.dispatchLayerStartDn);
        model_mod.setGpuLayerStartFa(&GpuAccelerator.dispatchLayerStartFa);
        model_mod.setGpuOprojFfnResident(&GpuAccelerator.dispatchOprojFfnResident);
        // Phase 4: fused final RMSNorm + LM head.
        model_mod.setGpuLmHead(&GpuAccelerator.dispatchLmHead);
    }

    std.debug.print("Generating (max {d} tokens)...\n", .{max_tokens});

    // Format with chat template
    const formatted = try Tokenizer.formatChatPrompt(allocator, null, prompt);
    defer allocator.free(formatted);

    // Tokenize
    const prompt_ids = try tokenizer.encode(allocator, formatted);
    defer allocator.free(prompt_ids);

    std.debug.print("Prompt: {d} tokens\n", .{prompt_ids.len});
    std.debug.print("Prompt IDs:", .{});
    for (prompt_ids) |id| std.debug.print(" {d}", .{id});
    std.debug.print("\n", .{});

    // Generate
    var timer = try std.time.Timer.start();
    const output_ids = try model.generate(prompt_ids, .{
        .max_tokens = max_tokens,
    });
    defer allocator.free(output_ids);
    const elapsed_ns = timer.read();

    std.debug.print("Output IDs:", .{});
    for (output_ids) |id| std.debug.print(" {d}", .{id});
    std.debug.print("\n", .{});

    // Decode and print
    const output_text = try tokenizer.decode(allocator, output_ids);
    defer allocator.free(output_text);

    std.debug.print("\n{s}\n", .{output_text});

    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const tok_per_sec = if (elapsed_ms > 0)
        @as(f64, @floatFromInt(output_ids.len)) / (elapsed_ms / 1000.0)
    else
        0.0;

    std.debug.print("\n--- {d} tokens in {d:.1}ms ({d:.1} tok/s) ---\n", .{
        output_ids.len,
        elapsed_ms,
        tok_per_sec,
    });
}

fn printUsage() void {
    std.debug.print(
        \\Tomoul Qwen3.5-0.8B — Hybrid DeltaNet + Attention Language Model
        \\
        \\Usage:
        \\  tomoul_qwen3_5-0.8b <command> [options]
        \\
        \\Commands:
        \\  generate <prompt>   Generate text from a prompt
        \\  version             Print version info
        \\
        \\Options:
        \\  -w, --weights <path>        Model weights (.tl file)
        \\  -t, --tokenizer <path>      Tokenizer vocabulary (.bin file)
        \\  -n, --max-tokens <N>        Maximum tokens to generate (default: 256)
        \\  --max-cache-len <N>         Maximum KV cache length (default: 4096)
        \\  --gpu                       Enable GPU acceleration (Vulkan)
        \\
    , .{});
}
