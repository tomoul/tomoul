const std = @import("std");
const registry = @import("src/models/registry.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ==========================================================================
    // Build Options
    // ==========================================================================
    // Filter: Allow user to build just ONE model
    // Usage: zig build -Dmodel=silero_vad
    const filter = b.option([]const u8, "model", "Build only a specific model name");

    // Bundled: Embed model weights in the library (for self-contained binaries)
    // Usage: zig build lib -Dbundled=true
    const bundled = b.option(bool, "bundled", "Embed model weights in native libraries") orelse false;

    // BLAS: Link OpenBLAS for accelerated matrix operations
    // Usage: zig build -Dmodel=whisper-tiny -Dblas=true
    // Requires OpenBLAS to be installed (apt install libopenblas-dev)
    const use_blas = b.option(bool, "blas", "Link OpenBLAS for accelerated matrix operations") orelse false;

    // zblas: Use pure Zig zblas library (default when not using OpenBLAS)
    // Usage: zig build -Dmodel=whisper-tiny -Dzblas=true (or just omit -Dblas)
    // No external dependencies required - works everywhere including WASM
    // Set -Dzblas=false -Dblas=false to use pure Zig fallback in ops.zig (for benchmarking)
    const use_zblas = b.option(bool, "zblas", "Use pure Zig zblas for matrix operations") orelse !use_blas;

    // LTO: Enable Link-Time Optimization for better cross-module inlining
    // Usage: zig build -Dmodel=whisper-tiny -Dzblas=true -Dlto=true
    const use_lto = b.option(bool, "lto", "Enable Link-Time Optimization") orelse false;

    // ==========================================================================
    // Main executable (model-specific or generic demo)
    // ==========================================================================
    const exe_name = if (filter) |model_name| b.fmt("tomoul_{s}", .{model_name}) else "tomoul";

    // Try to find model in registry to get module path
    var model_module_path: ?[]const u8 = null;
    if (filter) |model_name| {
        for (registry.models) |model_config| {
            if (std.mem.eql(u8, model_config.name, model_name)) {
                model_module_path = model_config.model_module;
                break;
            }
        }
    }

    const exe_source = if (filter) |model_name| blk: {
        // Derive CLI path from model module path
        const base_path = if (model_module_path) |path| blk2: {
            // Replace model.zig with cli.zig
            const model_zig = "/model.zig";
            if (std.mem.endsWith(u8, path, model_zig)) {
                const dir_path = path[0 .. path.len - model_zig.len];
                break :blk2 b.fmt("{s}/cli.zig", .{dir_path});
            }
            break :blk2 null;
        } else blk3: {
            // Fallback: derive from model name
            const model_name_underscore = b.allocator.alloc(u8, model_name.len) catch break :blk b.path("src/main.zig");
            @memcpy(model_name_underscore, model_name);
            for (model_name_underscore) |*c| {
                if (c.* == '-') c.* = '_';
            }
            break :blk3 b.fmt("src/models/{s}/cli.zig", .{model_name_underscore});
        };

        // Check if CLI exists
        if (base_path) |cli_path| {
            const cli_file = std.fs.cwd().openFile(cli_path, .{}) catch break :blk b.path("src/main.zig");
            cli_file.close();
            break :blk b.path(cli_path);
        }
        break :blk b.path("src/main.zig");
    } else b.path("src/main.zig");

    // Create core modules for native executable
    const tensor_module = b.createModule(.{
        .root_source_file = b.path("src/core/tensor.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Create build options for BLAS support in ops module
    const ops_options = b.addOptions();
    ops_options.addOption(bool, "use_blas", use_blas);
    ops_options.addOption(bool, "use_zblas", use_zblas and !use_blas);

    const ops_module = b.createModule(.{
        .root_source_file = b.path("src/core/ops.zig"),
        .target = target,
        .optimize = optimize,
    });
    ops_module.addImport("tensor.zig", tensor_module);
    ops_module.addOptions("build_options", ops_options);

    // Create zblas module from external dependency (pure Zig, no external deps)
    // Always create it but only import when use_zblas is true
    const zblas_dep = b.dependency("zblas", .{
        .target = target,
        .optimize = optimize,
    });
    const zblas_module = zblas_dep.module("zblas");
    if (use_zblas and !use_blas) {
        ops_module.addImport("zblas", zblas_module);
    }

    // Create BLAS module and link OpenBLAS if enabled
    if (use_blas) {
        const blas_module = b.createModule(.{
            .root_source_file = b.path("src/core/blas.zig"),
            .target = target,
            .optimize = optimize,
        });
        blas_module.linkSystemLibrary("openblas", .{});
        blas_module.link_libc = true;
        ops_module.addImport("blas.zig", blas_module);
    }

    const quantization_module = b.createModule(.{
        .root_source_file = b.path("src/core/quantization.zig"),
        .target = target,
        .optimize = optimize,
    });
    quantization_module.addImport("tensor.zig", tensor_module);
    quantization_module.addImport("ops.zig", ops_module);
    if (use_zblas and !use_blas) {
        quantization_module.addImport("zblas", zblas_module);
    }

    const loader_module = b.createModule(.{
        .root_source_file = b.path("src/core/loader.zig"),
        .target = target,
        .optimize = optimize,
    });
    loader_module.addImport("tensor.zig", tensor_module);
    loader_module.addImport("quantization.zig", quantization_module);
    loader_module.addImport("ops.zig", ops_module);

    // Attention module (supports F32, Q8, Q4, Q8_K via comptime generics)
    const attention_module = b.createModule(.{
        .root_source_file = b.path("src/core/attention.zig"),
        .target = target,
        .optimize = optimize,
    });
    attention_module.addImport("tensor.zig", tensor_module);
    attention_module.addImport("ops.zig", ops_module);
    attention_module.addImport("quantization.zig", quantization_module);

    // Transformer module (supports F32, Q8, Q4, Q8_K via comptime generics)
    const transformer_module = b.createModule(.{
        .root_source_file = b.path("src/core/transformer.zig"),
        .target = target,
        .optimize = optimize,
    });
    transformer_module.addImport("tensor.zig", tensor_module);
    transformer_module.addImport("ops.zig", ops_module);
    transformer_module.addImport("quantization.zig", quantization_module);
    transformer_module.addImport("attention.zig", attention_module);

    // Cache module (KV cache for autoregressive decoding)
    const cache_module = b.createModule(.{
        .root_source_file = b.path("src/core/cache.zig"),
        .target = target,
        .optimize = optimize,
    });
    cache_module.addImport("tensor.zig", tensor_module);

    // Audio module (mel spectrogram, FFT, audio loading - pure Zig)
    const audio_module = b.createModule(.{
        .root_source_file = b.path("src/core/audio.zig"),
        .target = target,
        .optimize = optimize,
    });
    audio_module.addImport("tensor.zig", tensor_module);

    // Create build options for bundled mode
    const exe_options = b.addOptions();
    exe_options.addOption(bool, "bundled", bundled);

    // If bundled, read and embed the model weights
    var embedded_weights: ?[]const u8 = null;
    if (bundled and filter != null) {
        for (registry.models) |model_config| {
            if (std.mem.eql(u8, model_config.name, filter.?) and model_config.supports_bundled) {
                const weights_path = model_config.weights_path orelse deriveWeightsPath(b, model_config.name);
                // Read the weights file at build time
                const weights_file = std.fs.cwd().readFileAlloc(
                    b.allocator,
                    weights_path,
                    100 * 1024 * 1024, // 100MB max
                ) catch |err| {
                    std.debug.print("Failed to read weights file {s}: {}\n", .{ weights_path, err });
                    break;
                };
                embedded_weights = weights_file;
                break;
            }
        }
    }
    exe_options.addOption(?[]const u8, "embedded_weights", embedded_weights);

    const exe_module = b.createModule(.{
        .root_source_file = exe_source,
        .target = target,
        .optimize = optimize,
    });
    exe_module.addImport("tensor.zig", tensor_module);
    exe_module.addImport("ops.zig", ops_module);
    exe_module.addImport("loader.zig", loader_module);
    exe_module.addImport("quantization.zig", quantization_module);
    exe_module.addImport("attention.zig", attention_module);
    exe_module.addImport("transformer.zig", transformer_module);
    exe_module.addImport("audio.zig", audio_module);
    exe_module.addOptions("build_options", exe_options);

    // Add model-specific imports if building for a specific model
    if (filter != null and model_module_path != null) {
        const model_path = model_module_path.?;
        const model_module = b.createModule(.{
            .root_source_file = b.path(model_path),
            .target = target,
            .optimize = optimize,
        });
        model_module.addImport("tensor.zig", tensor_module);
        model_module.addImport("ops.zig", ops_module);
        model_module.addImport("loader.zig", loader_module);
        model_module.addImport("quantization.zig", quantization_module);
        model_module.addImport("attention.zig", attention_module);
        model_module.addImport("transformer.zig", transformer_module);
        model_module.addImport("cache.zig", cache_module);
        model_module.addImport("audio.zig", audio_module);
        exe_module.addImport("model.zig", model_module);
    }

    const exe = b.addExecutable(.{
        .name = exe_name,
        .root_module = exe_module,
    });

    // Enable LTO if requested
    if (use_lto) {
        exe.want_lto = true;
    }

    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the Tomoul inference engine");
    run_step.dependOn(&run_cmd.step);

    // ==========================================================================
    // Whisper HTTP Server
    // ==========================================================================
    if (filter) |model_name| {
        if (std.mem.eql(u8, model_name, "whisper-tiny") or
            std.mem.eql(u8, model_name, "whisper-base") or
            std.mem.eql(u8, model_name, "whisper-small") or
            std.mem.eql(u8, model_name, "whisper-medium") or
            std.mem.eql(u8, model_name, "whisper-large"))
        {
            const server_module = b.createModule(.{
                .root_source_file = b.path("src/models/whisper/server.zig"),
                .target = target,
                .optimize = optimize,
            });
            server_module.addImport("tensor.zig", tensor_module);
            server_module.addImport("ops.zig", ops_module);
            server_module.addImport("loader.zig", loader_module);
            server_module.addImport("quantization.zig", quantization_module);
            server_module.addImport("attention.zig", attention_module);
            server_module.addImport("transformer.zig", transformer_module);
            server_module.addImport("audio.zig", audio_module);

            // Add model module for whisper
            if (model_module_path) |path| {
                const whisper_model_module = b.createModule(.{
                    .root_source_file = b.path(path),
                    .target = target,
                    .optimize = optimize,
                });
                whisper_model_module.addImport("tensor.zig", tensor_module);
                whisper_model_module.addImport("ops.zig", ops_module);
                whisper_model_module.addImport("loader.zig", loader_module);
                whisper_model_module.addImport("quantization.zig", quantization_module);
                whisper_model_module.addImport("attention.zig", attention_module);
                whisper_model_module.addImport("transformer.zig", transformer_module);
                whisper_model_module.addImport("cache.zig", cache_module);
                whisper_model_module.addImport("audio.zig", audio_module);
                server_module.addImport("model.zig", whisper_model_module);
            }

            const server_exe = b.addExecutable(.{
                .name = "whisper-server",
                .root_module = server_module,
            });

            b.installArtifact(server_exe);

            const server_run_cmd = b.addRunArtifact(server_exe);
            server_run_cmd.step.dependOn(b.getInstallStep());
            if (b.args) |args| {
                server_run_cmd.addArgs(args);
            }

            const server_step = b.step("whisper-server", "Build and run the Whisper HTTP server");
            server_step.dependOn(&server_run_cmd.step);
        }
    }

    // ==========================================================================
    // Unit tests
    // ==========================================================================
    const test_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("tensor.zig", tensor_module);
    test_module.addImport("ops.zig", ops_module);
    test_module.addImport("loader.zig", loader_module);
    test_module.addImport("quantization.zig", quantization_module);

    const unit_tests = b.addTest(.{
        .root_module = test_module,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // ==========================================================================
    // Integration tests
    // ==========================================================================
    const tomoul_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    tomoul_module.addImport("tensor.zig", tensor_module);
    tomoul_module.addImport("ops.zig", ops_module);
    tomoul_module.addImport("loader.zig", loader_module);
    tomoul_module.addImport("quantization.zig", quantization_module);

    const integration_module = b.createModule(.{
        .root_source_file = b.path("tests/integration.zig"),
        .target = target,
        .optimize = optimize,
    });
    integration_module.addImport("tomoul", tomoul_module);

    const integration_tests = b.addTest(.{
        .root_module = integration_module,
    });

    const run_integration_tests = b.addRunArtifact(integration_tests);
    const integration_test_step = b.step("test-integration", "Run integration tests with fixtures");
    integration_test_step.dependOn(&run_integration_tests.step);

    // ==========================================================================
    // Sentence Transformer validation tests
    // ==========================================================================
    const st_tokenizer_module = b.createModule(.{
        .root_source_file = b.path("src/models/sentence_transformer/tokenizer.zig"),
        .target = target,
        .optimize = optimize,
    });

    const st_model_module = b.createModule(.{
        .root_source_file = b.path("src/models/sentence_transformer/model.zig"),
        .target = target,
        .optimize = optimize,
    });
    st_model_module.addImport("tensor.zig", tensor_module);
    st_model_module.addImport("ops.zig", ops_module);
    st_model_module.addImport("loader.zig", loader_module);
    st_model_module.addImport("quantization.zig", quantization_module);
    st_model_module.addImport("attention.zig", attention_module);
    st_model_module.addImport("transformer.zig", transformer_module);
    st_model_module.addImport("cache.zig", cache_module);
    st_model_module.addImport("audio.zig", audio_module);
    st_model_module.addImport("tokenizer.zig", st_tokenizer_module);

    // Export for downstream consumers (e.g., habor CLI)
    b.modules.put("sentence_transformer", st_model_module) catch @panic("OOM");

    const st_test_module = b.createModule(.{
        .root_source_file = b.path("tests/test_sentence_transformer.zig"),
        .target = target,
        .optimize = optimize,
    });
    st_test_module.addImport("model", st_model_module);
    st_test_module.addImport("tokenizer", st_tokenizer_module);

    const st_tests = b.addTest(.{
        .root_module = st_test_module,
    });

    const run_st_tests = b.addRunArtifact(st_tests);
    const st_test_step = b.step("test-sentence-transformer", "Run sentence transformer validation tests");
    st_test_step.dependOn(&run_st_tests.step);

    // ==========================================================================
    // Qwen3.5 unit tests
    // ==========================================================================
    const qwen35_test_module = b.createModule(.{
        .root_source_file = b.path("tests/test_qwen3_5.zig"),
        .target = target,
        .optimize = optimize,
    });
    qwen35_test_module.addImport("tensor.zig", tensor_module);
    qwen35_test_module.addImport("ops.zig", ops_module);

    const qwen35_tests = b.addTest(.{
        .root_module = qwen35_test_module,
    });

    const run_qwen35_tests = b.addRunArtifact(qwen35_tests);
    const qwen35_test_step = b.step("test-qwen3_5", "Run Qwen3.5 model unit tests");
    qwen35_test_step.dependOn(&run_qwen35_tests.step);

    // ==========================================================================
    // GPU HAL (Hardware Abstraction Layer) + Vulkan Backend
    // ==========================================================================
    {
        // --- Runtime Vulkan loader (dlopen — zero link-time dependency) ---
        const vk_loader_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/vk_loader.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true, // needed for dlopen to search standard library paths
        });

        // --- Low-level Vulkan wrapper (uses vk_loader at runtime) ---
        const vulkan_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/vulkan.zig"),
            .target = target,
            .optimize = optimize,
        });
        vulkan_module.addImport("vk_loader", vk_loader_module);

        // --- Vulkan forward pass (embeds SPIR-V shaders via @embedFile) ---
        const vulkan_forward_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/vulkan_forward.zig"),
            .target = target,
            .optimize = optimize,
        });
        vulkan_forward_module.addImport("vulkan", vulkan_module);

        // --- Qwen3.5 GPU accelerator (SGEMV matvec via Vulkan compute) ---
        const qwen3_5_gpu_module = b.createModule(.{
            .root_source_file = b.path("src/models/qwen3_5/qwen3_5_gpu.zig"),
            .target = target,
            .optimize = optimize,
        });
        qwen3_5_gpu_module.addImport("vulkan", vulkan_module);
        qwen3_5_gpu_module.addImport("vk_loader", vk_loader_module);
        exe_module.addImport("qwen3_5_gpu", qwen3_5_gpu_module);

        // --- HAL interface (backend-agnostic) ---
        const hal_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/hal.zig"),
            .target = target,
            .optimize = optimize,
        });

        // --- Vulkan backend (implements HAL for Linux/Windows/Android) ---
        const vulkan_backend_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/vulkan_backend.zig"),
            .target = target,
            .optimize = optimize,
        });
        vulkan_backend_module.addImport("vulkan_forward", vulkan_forward_module);

        // --- Metal low-level wrapper (Obj-C runtime, zero link-time Metal dep) ---
        const metal_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/metal.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true, // needed for dlopen + libobjc
        });
        // Only link libobjc on Apple platforms (Metal is dead code elsewhere)
        const os_tag = target.query.os_tag orelse @import("builtin").os.tag;
        if (os_tag == .macos or os_tag == .ios) {
            metal_module.linkSystemLibrary("objc", .{}); // for objc_msgSend, sel_registerName, objc_getClass
        }

        // --- Metal forward pass (embeds MSL shaders via @embedFile) ---
        const metal_forward_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/metal_forward.zig"),
            .target = target,
            .optimize = optimize,
        });
        metal_forward_module.addImport("metal", metal_module);

        // --- Metal backend (implements HAL for macOS/iOS) ---
        const metal_backend_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/metal_backend.zig"),
            .target = target,
            .optimize = optimize,
        });
        metal_backend_module.addImport("metal_forward", metal_forward_module);

        // --- WebGPU low-level bridge (extern JS imports for WASM) ---
        const webgpu_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/webgpu.zig"),
            .target = target,
            .optimize = optimize,
        });

        // --- WebGPU forward pass (embeds WGSL shaders via @embedFile) ---
        const webgpu_forward_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/webgpu_forward.zig"),
            .target = target,
            .optimize = optimize,
        });
        webgpu_forward_module.addImport("webgpu", webgpu_module);

        // --- WebGPU backend (implements HAL for WASM/browser) ---
        const webgpu_backend_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/webgpu_backend.zig"),
            .target = target,
            .optimize = optimize,
        });
        webgpu_backend_module.addImport("webgpu_forward", webgpu_forward_module);

        // HAL imports all backends for auto-detection
        hal_module.addImport("vulkan_backend", vulkan_backend_module);
        hal_module.addImport("vulkan_forward", vulkan_forward_module);
        hal_module.addImport("metal_backend", metal_backend_module);
        hal_module.addImport("metal_forward", metal_forward_module);
        hal_module.addImport("webgpu_backend", webgpu_backend_module);
        hal_module.addImport("webgpu_forward", webgpu_forward_module);

        // Basic Vulkan compute tests (vec_add, sgemm)
        const gpu_test_module = b.createModule(.{
            .root_source_file = b.path("tests/test_vulkan_compute.zig"),
            .target = target,
            .optimize = optimize,
        });
        gpu_test_module.addImport("vulkan", vulkan_module);

        const gpu_tests = b.addTest(.{
            .root_module = gpu_test_module,
        });

        const run_gpu_tests = b.addRunArtifact(gpu_tests);
        const gpu_test_step = b.step("test-gpu", "Run Vulkan GPU compute tests");
        gpu_test_step.dependOn(&run_gpu_tests.step);

        // GPU forward pass tests (layernorm, gelu, attention, full forward)
        const gpu_fwd_test_module = b.createModule(.{
            .root_source_file = b.path("tests/test_gpu_forward.zig"),
            .target = target,
            .optimize = optimize,
        });
        gpu_fwd_test_module.addImport("vulkan", vulkan_module);
        gpu_fwd_test_module.addImport("vulkan_forward", vulkan_forward_module);

        const gpu_fwd_tests = b.addTest(.{
            .root_module = gpu_fwd_test_module,
        });

        const run_gpu_fwd_tests = b.addRunArtifact(gpu_fwd_tests);
        const gpu_fwd_test_step = b.step("test-gpu-forward", "Run GPU forward pass tests");
        gpu_fwd_test_step.dependOn(&run_gpu_fwd_tests.step);
        gpu_test_step.dependOn(&run_gpu_fwd_tests.step);

        // GPU model wrapper (uses HAL auto-detection)
        const gpu_model_module = b.createModule(.{
            .root_source_file = b.path("src/gpu/gpu_model.zig"),
            .target = target,
            .optimize = optimize,
        });
        gpu_model_module.addImport("hal", hal_module);
        gpu_model_module.addImport("model", st_model_module);
        gpu_model_module.addImport("tokenizer", st_tokenizer_module);
        gpu_model_module.addImport("transformer", transformer_module);

        // GPU vs CPU model test (loads real weights)
        const gpu_model_test_module = b.createModule(.{
            .root_source_file = b.path("tests/test_gpu_model.zig"),
            .target = target,
            .optimize = optimize,
        });
        gpu_model_test_module.addImport("gpu_model", gpu_model_module);
        gpu_model_test_module.addImport("model", st_model_module);
        gpu_model_test_module.addImport("tokenizer", st_tokenizer_module);

        const gpu_model_tests = b.addTest(.{
            .root_module = gpu_model_test_module,
        });

        const run_gpu_model_tests = b.addRunArtifact(gpu_model_tests);
        const gpu_model_test_step = b.step("test-gpu-model", "Run GPU vs CPU model comparison tests");
        gpu_model_test_step.dependOn(&run_gpu_model_tests.step);
    }

    // ==========================================================================
    // WebAssembly targets (data-driven from model_registry.zig)
    // ==========================================================================
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    // Create shared core modules for Wasm (used by all models)
    const wasm_tensor_module = b.createModule(.{
        .root_source_file = b.path("src/core/tensor.zig"),
        .target = wasm_target,
        .optimize = .ReleaseFast,
    });

    // WASM build options - disable zblas (LLVM 20 x64 backend crashes with @Vector on wasm32)
    // The zblas source is wasm32-correct but LLVM's x64-hosted wasm codegen has a bug.
    // WASM uses the scalar fallback in ops.zig until LLVM fixes the issue.
    const wasm_ops_options = b.addOptions();
    wasm_ops_options.addOption(bool, "use_blas", false);
    wasm_ops_options.addOption(bool, "use_zblas", false);

    const wasm_ops_module = b.createModule(.{
        .root_source_file = b.path("src/core/ops.zig"),
        .target = wasm_target,
        .optimize = .ReleaseFast,
    });
    wasm_ops_module.addImport("tensor.zig", wasm_tensor_module);
    wasm_ops_module.addOptions("build_options", wasm_ops_options);

    const wasm_quantization_module = b.createModule(.{
        .root_source_file = b.path("src/core/quantization.zig"),
        .target = wasm_target,
        .optimize = .ReleaseFast,
    });
    wasm_quantization_module.addImport("tensor.zig", wasm_tensor_module);
    wasm_quantization_module.addImport("ops.zig", wasm_ops_module);

    const wasm_loader_module = b.createModule(.{
        .root_source_file = b.path("src/core/loader.zig"),
        .target = wasm_target,
        .optimize = .ReleaseFast,
    });
    wasm_loader_module.addImport("tensor.zig", wasm_tensor_module);
    wasm_loader_module.addImport("quantization.zig", wasm_quantization_module);

    // Generic attention module (supports F32, Q8, Q4, Q8_K) - WASM
    const wasm_attention_module = b.createModule(.{
        .root_source_file = b.path("src/core/attention.zig"),
        .target = wasm_target,
        .optimize = .ReleaseFast,
    });
    wasm_attention_module.addImport("tensor.zig", wasm_tensor_module);
    wasm_attention_module.addImport("ops.zig", wasm_ops_module);
    wasm_attention_module.addImport("quantization.zig", wasm_quantization_module);

    // Generic transformer module (supports F32, Q8, Q4, Q8_K) - WASM
    const wasm_transformer_module = b.createModule(.{
        .root_source_file = b.path("src/core/transformer.zig"),
        .target = wasm_target,
        .optimize = .ReleaseFast,
    });
    wasm_transformer_module.addImport("tensor.zig", wasm_tensor_module);
    wasm_transformer_module.addImport("ops.zig", wasm_ops_module);
    wasm_transformer_module.addImport("quantization.zig", wasm_quantization_module);
    wasm_transformer_module.addImport("attention.zig", wasm_attention_module);

    // Master wasm step that builds all models
    const wasm_step = b.step("wasm", "Build all WebAssembly targets");

    // Iterate through all registered models
    for (registry.models) |model| {
        // Skip if filter is set and names don't match
        if (filter) |f| {
            if (!std.mem.eql(u8, f, model.name)) continue;
        }

        // Skip WASM build for models that don't support bundled weights
        // (WASM requires embedded weights, can't load at runtime)
        if (!model.supports_bundled) {
            continue;
        }

        // Build wasm target for this model
        buildWasmModel(
            b,
            model,
            wasm_target,
            wasm_tensor_module,
            wasm_ops_module,
            wasm_loader_module,
            wasm_quantization_module,
            wasm_attention_module,
            wasm_transformer_module,
            wasm_step,
        );
    }

    // ==========================================================================
    // Native C Library (static + shared with auto-generated header)
    // ==========================================================================
    const lib_step = b.step("lib", "Build native C libraries with headers");

    for (registry.models) |model| {
        // Skip if filter is set and names don't match
        if (filter) |f| {
            if (!std.mem.eql(u8, f, model.name)) continue;
        }

        buildNativeLib(b, model, target, optimize, bundled, lib_step);
    }
}

// ==========================================================================
// Path derivation helpers (convention over configuration)
// ==========================================================================
// Given model name "silero_vad", derive:
//   - wasm_binding  → src/models/silero_vad/wasm.zig
//   - c_binding     → src/models/silero_vad/c.zig
//   - model_module  → src/models/silero_vad/model.zig
//   - weights_path  → models/silero_vad.tl
//   - example_dir   → examples/silero-vad (underscores → dashes)

fn deriveWasmBinding(b: *std.Build, name: []const u8) []const u8 {
    return b.fmt("src/models/{s}/wasm.zig", .{name});
}

fn deriveCBinding(b: *std.Build, name: []const u8) []const u8 {
    return b.fmt("src/models/{s}/c.zig", .{name});
}

fn deriveModelModule(b: *std.Build, name: []const u8) []const u8 {
    return b.fmt("src/models/{s}/model.zig", .{name});
}

fn deriveWeightsPath(b: *std.Build, name: []const u8) []const u8 {
    return b.fmt("artifacts/{s}.tl", .{name});
}

fn deriveExampleDir(b: *std.Build, name: []const u8) []const u8 {
    // Convert underscores to dashes for example directory
    var dashed: [256]u8 = undefined;
    var len: usize = 0;
    for (name) |c| {
        if (len >= dashed.len) break;
        dashed[len] = if (c == '_') '-' else c;
        len += 1;
    }
    return b.fmt("examples/{s}", .{dashed[0..len]});
}

/// Build a WebAssembly target for a specific model
fn buildWasmModel(
    b: *std.Build,
    model: registry.ModelConfig,
    wasm_target: std.Build.ResolvedTarget,
    wasm_tensor_module: *std.Build.Module,
    wasm_ops_module: *std.Build.Module,
    wasm_loader_module: *std.Build.Module,
    wasm_quantization_module: *std.Build.Module,
    wasm_attention_module: *std.Build.Module,
    wasm_transformer_module: *std.Build.Module,
    wasm_step: *std.Build.Step,
) void {
    // Use explicit paths from registry, or derive from model name (convention over configuration)
    const model_module_path = model.model_module orelse deriveModelModule(b, model.name);
    const wasm_binding_path = model.wasm_binding orelse deriveWasmBinding(b, model.name);
    const weights_path = model.weights_path orelse deriveWeightsPath(b, model.name);

    // Create the model module for Wasm
    const wasm_model_module = b.createModule(.{
        .root_source_file = b.path(model_module_path),
        .target = wasm_target,
        .optimize = .ReleaseFast,
    });
    wasm_model_module.addImport("tensor.zig", wasm_tensor_module);
    wasm_model_module.addImport("ops.zig", wasm_ops_module);
    wasm_model_module.addImport("loader.zig", wasm_loader_module);
    wasm_model_module.addImport("quantization.zig", wasm_quantization_module);
    wasm_model_module.addImport("attention.zig", wasm_attention_module);
    wasm_model_module.addImport("transformer.zig", wasm_transformer_module);

    // Create the binding module
    const wasm_binding = b.createModule(.{
        .root_source_file = b.path(wasm_binding_path),
        .target = wasm_target,
        .optimize = .ReleaseFast,
    });
    wasm_binding.addImport("tensor", wasm_tensor_module);
    wasm_binding.addImport("model", wasm_model_module);

    // =========================================================================
    // GPU modules for WASM (WebGPU acceleration via JS bridge)
    // =========================================================================

    // Tokenizer module (needed by gpu_model)
    const wasm_tokenizer_module = b.createModule(.{
        .root_source_file = b.path("src/models/sentence_transformer/tokenizer.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    wasm_model_module.addImport("tokenizer.zig", wasm_tokenizer_module);

    // WebGPU modules (active on wasm32)
    // Note: Vulkan and Metal modules are NOT built for wasm32 — they contain
    // platform-specific code (objc_msgSend @ptrCast, Vulkan loaders) that
    // triggers LLVM Invalid Cast errors on 32-bit targets (Zig issue #24345).
    const wasm_webgpu_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/webgpu.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    const wasm_webgpu_forward_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/webgpu_forward.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    wasm_webgpu_forward_module.addImport("webgpu", wasm_webgpu_module);

    const wasm_webgpu_backend_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/webgpu_backend.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    wasm_webgpu_backend_module.addImport("webgpu_forward", wasm_webgpu_forward_module);

    // HAL interface (auto-detects WebGPU on wasm32-freestanding)
    const wasm_hal_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/hal.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    wasm_hal_module.addImport("webgpu_backend", wasm_webgpu_backend_module);
    wasm_hal_module.addImport("webgpu_forward", wasm_webgpu_forward_module);

    // GPU model wrapper (uses HAL auto-detection)
    const wasm_gpu_model_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/gpu_model.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    wasm_gpu_model_module.addImport("hal", wasm_hal_module);
    wasm_gpu_model_module.addImport("model", wasm_model_module);
    wasm_gpu_model_module.addImport("tokenizer", wasm_tokenizer_module);
    wasm_gpu_model_module.addImport("transformer", wasm_transformer_module);

    wasm_binding.addImport("gpu_model", wasm_gpu_model_module);

    // Embed model weights if bundled
    // For WASM, prefer Q8K variant if available (smaller download, fits in less memory)
    if (model.supports_bundled) {
        const wasm_weights = blk: {
            for (model.weight_variants) |v| {
                if (std.mem.eql(u8, v.suffix, "-q8k")) break :blk v.path;
            }
            break :blk weights_path;
        };
        wasm_binding.addAnonymousImport("model_weights", .{
            .root_source_file = b.path(wasm_weights),
        });
    }

    // Embed vocab if provided
    if (model.vocab_path) |vp| {
        wasm_binding.addAnonymousImport("vocab", .{
            .root_source_file = b.path(vp),
        });
    }

    // Create the wasm executable
    const wasm_name = b.fmt("tomoul_{s}", .{model.name});
    const wasm = b.addExecutable(.{
        .name = wasm_name,
        .root_module = wasm_binding,
    });

    // Disable entry point (we use exported functions instead)
    wasm.entry = .disabled;

    // Export symbols for JavaScript access
    wasm.root_module.export_symbol_names = model.export_symbols;

    // Install to default location
    const wasm_install = b.addInstallArtifact(wasm, .{});
    wasm_step.dependOn(&wasm_install.step);

    // Also copy to example directory if has_example is true
    if (model.has_example) {
        const example_dir = deriveExampleDir(b, model.name);
        const wasm_filename = b.fmt("tomoul_{s}.wasm", .{model.name});
        const install_path = b.fmt("../{s}", .{example_dir});
        const wasm_copy = b.addInstallFileWithDir(
            wasm.getEmittedBin(),
            .{ .custom = install_path },
            wasm_filename,
        );
        wasm_step.dependOn(&wasm_copy.step);
    }
}

/// Build native C library (static + shared) with auto-generated header
/// When bundled=true, model weights are embedded via @embedFile
fn buildNativeLib(
    b: *std.Build,
    model: registry.ModelConfig,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    bundled: bool,
    lib_step: *std.Build.Step,
) void {
    // Use explicit paths from registry, or derive from model name (convention over configuration)
    const c_binding_path = model.c_binding orelse deriveCBinding(b, model.name);
    const model_module_path = model.model_module orelse deriveModelModule(b, model.name);
    const weights_path = model.weights_path orelse deriveWeightsPath(b, model.name);

    // Create shared modules for native builds
    const tensor_module = b.createModule(.{
        .root_source_file = b.path("src/core/tensor.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Create ops build options (use zblas for lib builds - no external deps)
    const ops_build_options = b.addOptions();
    ops_build_options.addOption(bool, "use_blas", false);
    ops_build_options.addOption(bool, "use_zblas", true);

    const ops_module = b.createModule(.{
        .root_source_file = b.path("src/core/ops.zig"),
        .target = target,
        .optimize = optimize,
    });
    ops_module.addImport("tensor.zig", tensor_module);
    ops_module.addOptions("build_options", ops_build_options);

    // zblas for native lib builds (pure Zig - no external dependencies)
    const lib_zblas_dep = b.dependency("zblas", .{
        .target = target,
        .optimize = optimize,
    });
    ops_module.addImport("zblas", lib_zblas_dep.module("zblas"));

    const quantization_module = b.createModule(.{
        .root_source_file = b.path("src/core/quantization.zig"),
        .target = target,
        .optimize = optimize,
    });
    quantization_module.addImport("tensor.zig", tensor_module);
    quantization_module.addImport("ops.zig", ops_module);
    quantization_module.addImport("zblas", lib_zblas_dep.module("zblas"));

    const loader_module = b.createModule(.{
        .root_source_file = b.path("src/core/loader.zig"),
        .target = target,
        .optimize = optimize,
    });
    loader_module.addImport("tensor.zig", tensor_module);
    loader_module.addImport("quantization.zig", quantization_module);
    loader_module.addImport("ops.zig", ops_module);

    // Generic attention module (supports F32, Q8, Q4, Q8_K) - Native
    const attention_module = b.createModule(.{
        .root_source_file = b.path("src/core/attention.zig"),
        .target = target,
        .optimize = optimize,
    });
    attention_module.addImport("tensor.zig", tensor_module);
    attention_module.addImport("ops.zig", ops_module);
    attention_module.addImport("quantization.zig", quantization_module);

    // Generic transformer module (supports F32, Q8, Q4, Q8_K) - Native
    const transformer_module = b.createModule(.{
        .root_source_file = b.path("src/core/transformer.zig"),
        .target = target,
        .optimize = optimize,
    });
    transformer_module.addImport("tensor.zig", tensor_module);
    transformer_module.addImport("ops.zig", ops_module);
    transformer_module.addImport("quantization.zig", quantization_module);
    transformer_module.addImport("attention.zig", attention_module);

    // Audio module (mel spectrogram, FFT, audio loading - pure Zig) - Native lib
    const audio_module = b.createModule(.{
        .root_source_file = b.path("src/core/audio.zig"),
        .target = target,
        .optimize = optimize,
    });
    audio_module.addImport("tensor.zig", tensor_module);

    // Create the model module
    const model_module = b.createModule(.{
        .root_source_file = b.path(model_module_path),
        .target = target,
        .optimize = optimize,
    });
    model_module.addImport("tensor.zig", tensor_module);
    model_module.addImport("ops.zig", ops_module);
    model_module.addImport("loader.zig", loader_module);
    model_module.addImport("quantization.zig", quantization_module);
    model_module.addImport("attention.zig", attention_module);
    model_module.addImport("transformer.zig", transformer_module);
    model_module.addImport("audio.zig", audio_module);

    // Create build options for bundled/lite mode
    const options = b.addOptions();
    options.addOption(bool, "embed_weights", bundled and model.supports_bundled);

    // =========================================================================
    // GPU modules for native lib (Vulkan/Metal acceleration)
    // =========================================================================

    // Tokenizer module (needed by model.zig as @import("tokenizer.zig") and by gpu_model separately)
    const lib_tokenizer_module = b.createModule(.{
        .root_source_file = b.path("src/models/sentence_transformer/tokenizer.zig"),
        .target = target,
        .optimize = optimize,
    });
    // Wire tokenizer into model module so model.zig's @import("tokenizer.zig") resolves to the module
    model_module.addImport("tokenizer.zig", lib_tokenizer_module);

    // Runtime Vulkan loader (dlopen — zero link-time dependency)
    const lib_vk_loader_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/vk_loader.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // Low-level Vulkan wrapper
    const lib_vulkan_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/vulkan.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_vulkan_module.addImport("vk_loader", lib_vk_loader_module);

    // Vulkan forward pass (embeds SPIR-V shaders)
    const lib_vulkan_forward_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/vulkan_forward.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_vulkan_forward_module.addImport("vulkan", lib_vulkan_module);

    // Vulkan backend (implements HAL)
    const lib_vulkan_backend_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/vulkan_backend.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_vulkan_backend_module.addImport("vulkan_forward", lib_vulkan_forward_module);

    // --- Metal low-level wrapper (Obj-C runtime, zero link-time Metal dep) ---
    const lib_metal_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/metal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    // Only link libobjc on Apple platforms (Metal is dead code elsewhere)
    const lib_os_tag = target.query.os_tag orelse @import("builtin").os.tag;
    if (lib_os_tag == .macos or lib_os_tag == .ios) {
        lib_metal_module.linkSystemLibrary("objc", .{});
    }

    // --- Metal forward pass (embeds MSL shaders via @embedFile) ---
    const lib_metal_forward_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/metal_forward.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_metal_forward_module.addImport("metal", lib_metal_module);

    // --- Metal backend (implements HAL for macOS/iOS) ---
    const lib_metal_backend_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/metal_backend.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_metal_backend_module.addImport("metal_forward", lib_metal_forward_module);

    // --- WebGPU low-level bridge (extern JS imports for WASM) ---
    const lib_webgpu_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/webgpu.zig"),
        .target = target,
        .optimize = optimize,
    });

    // --- WebGPU forward pass (embeds WGSL shaders via @embedFile) ---
    const lib_webgpu_forward_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/webgpu_forward.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_webgpu_forward_module.addImport("webgpu", lib_webgpu_module);

    // --- WebGPU backend (implements HAL for WASM/browser) ---
    const lib_webgpu_backend_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/webgpu_backend.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_webgpu_backend_module.addImport("webgpu_forward", lib_webgpu_forward_module);

    // HAL interface (backend-agnostic)
    const lib_hal_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/hal.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_hal_module.addImport("vulkan_backend", lib_vulkan_backend_module);
    lib_hal_module.addImport("vulkan_forward", lib_vulkan_forward_module);
    lib_hal_module.addImport("metal_backend", lib_metal_backend_module);
    lib_hal_module.addImport("metal_forward", lib_metal_forward_module);
    lib_hal_module.addImport("webgpu_backend", lib_webgpu_backend_module);
    lib_hal_module.addImport("webgpu_forward", lib_webgpu_forward_module);

    // GPU model wrapper (uses HAL auto-detection)
    const lib_gpu_model_module = b.createModule(.{
        .root_source_file = b.path("src/gpu/gpu_model.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_gpu_model_module.addImport("hal", lib_hal_module);
    lib_gpu_model_module.addImport("model", model_module);
    lib_gpu_model_module.addImport("tokenizer", lib_tokenizer_module);
    lib_gpu_model_module.addImport("transformer", transformer_module);

    // Helper to configure a C binding module
    const configureBindingModule = struct {
        fn configure(
            builder: *std.Build,
            c_path: []const u8,
            tgt: std.Build.ResolvedTarget,
            opt: std.builtin.OptimizeMode,
            tensor_mod: *std.Build.Module,
            model_mod: *std.Build.Module,
            loader_mod: *std.Build.Module,
            opts: *std.Build.Step.Options,
            is_bundled: bool,
            supports_bundled: bool,
            w_path: []const u8,
        ) *std.Build.Module {
            const c_mod = builder.createModule(.{
                .root_source_file = builder.path(c_path),
                .target = tgt,
                .optimize = opt,
            });
            c_mod.addImport("tensor", tensor_mod);
            c_mod.addImport("model", model_mod);
            c_mod.addImport("loader", loader_mod); // For QuantFormat access
            c_mod.addOptions("build_options", opts);

            // Embed model weights if bundled mode
            if (is_bundled and supports_bundled) {
                c_mod.addAnonymousImport("model_weights", .{
                    .root_source_file = builder.path(w_path),
                });
            }

            return c_mod;
        }
    }.configure;

    // Create C binding module for static library
    const c_binding_module = configureBindingModule(
        b,
        c_binding_path,
        target,
        optimize,
        tensor_module,
        model_module,
        loader_module,
        options,
        bundled,
        model.supports_bundled,
        weights_path,
    );

    // Embed vocab if provided
    if (model.vocab_path) |vp| {
        c_binding_module.addAnonymousImport("vocab", .{
            .root_source_file = b.path(vp),
        });
    }

    // Add GPU module to static lib binding
    c_binding_module.addImport("gpu_model", lib_gpu_model_module);

    // Build static library
    const lib_name = b.fmt("tomoul_{s}", .{model.name});
    const static_lib = b.addLibrary(.{
        .linkage = .static,
        .name = lib_name,
        .root_module = c_binding_module,
    });

    // Install static library
    const static_install = b.addInstallArtifact(static_lib, .{});
    lib_step.dependOn(&static_install.step);

    // Note: C header generation is handled by scripts/release.py
    // which parses export functions from c.zig and generates proper C declarations

    // Create C binding module for shared library (modules can only be used once)
    const c_binding_module_shared = configureBindingModule(
        b,
        c_binding_path,
        target,
        optimize,
        tensor_module,
        model_module,
        loader_module,
        options,
        bundled,
        model.supports_bundled,
        weights_path,
    );

    // Embed vocab if provided
    if (model.vocab_path) |vp| {
        c_binding_module_shared.addAnonymousImport("vocab", .{
            .root_source_file = b.path(vp),
        });
    }

    // Add GPU module to shared lib binding
    c_binding_module_shared.addImport("gpu_model", lib_gpu_model_module);

    const shared_lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = lib_name,
        .root_module = c_binding_module_shared,
    });

    const shared_install = b.addInstallArtifact(shared_lib, .{});
    lib_step.dependOn(&shared_install.step);
}
