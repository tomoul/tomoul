const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main executable
    const exe = b.addExecutable(.{
        .name = "tomoul",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the Tomoul inference engine");
    run_step.dependOn(&run_cmd.step);

    // Unit tests
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Integration tests (uses fixtures from tests/fixtures/)
    // Create tomoul module from main.zig to access all exports
    const tomoul_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

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

    // WebAssembly build target (bundled - includes model weights)
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    // Create core modules for Wasm
    // ops.zig and loader.zig use relative imports (tensor.zig) so we need to
    // map "tensor.zig" to our tensor module for them to work
    const wasm_tensor_module = b.createModule(.{
        .root_source_file = b.path("src/core/tensor.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });

    const wasm_ops_module = b.createModule(.{
        .root_source_file = b.path("src/core/ops.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    wasm_ops_module.addImport("tensor.zig", wasm_tensor_module);

    const wasm_loader_module = b.createModule(.{
        .root_source_file = b.path("src/core/loader.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    wasm_loader_module.addImport("tensor.zig", wasm_tensor_module);

    // Create the SileroVAD model module for Wasm
    const wasm_vad_model = b.createModule(.{
        .root_source_file = b.path("src/models/silero_vad.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    // Map relative imports to our wasm modules
    wasm_vad_model.addImport("../core/tensor.zig", wasm_tensor_module);
    wasm_vad_model.addImport("../core/ops.zig", wasm_ops_module);
    wasm_vad_model.addImport("../core/loader.zig", wasm_loader_module);

    // The binding file imports the model and exposes it to JS
    const wasm_binding = b.createModule(.{
        .root_source_file = b.path("src/bindings/wasm_vad.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    wasm_binding.addImport("tensor", wasm_tensor_module);
    wasm_binding.addImport("vad", wasm_vad_model);
    // Embed model weights
    wasm_binding.addAnonymousImport("model_weights", .{
        .root_source_file = b.path("models/silero_vad.tl"),
    });

    const wasm = b.addExecutable(.{
        .name = "tomoul_vad",
        .root_module = wasm_binding,
    });

    // Disable entry point (we use exported functions instead)
    wasm.entry = .disabled;

    // Export symbols for JavaScript access
    wasm.root_module.export_symbol_names = &.{
        "init",
        "get_input_buffer_ptr",
        "get_max_input_samples",
        "process_audio",
        "reset_state",
        "is_ready",
        "get_version",
    };

    const wasm_step = b.step("wasm", "Build WebAssembly target (bundled with model)");
    wasm_step.dependOn(&b.addInstallArtifact(wasm, .{}).step);
}
