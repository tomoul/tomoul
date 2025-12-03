const std = @import("std");
const registry = @import("model_registry.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ==========================================================================
    // Filter: Allow user to build just ONE model
    // Usage: zig build -Dmodel=silero_vad
    // ==========================================================================
    const filter = b.option([]const u8, "model", "Build only a specific model name");

    // ==========================================================================
    // Main executable
    // ==========================================================================
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

    // ==========================================================================
    // Unit tests
    // ==========================================================================
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

    // ==========================================================================
    // Integration tests
    // ==========================================================================
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

    // Master wasm step that builds all models
    const wasm_step = b.step("wasm", "Build all WebAssembly targets");

    // Iterate through all registered models
    for (registry.models) |model| {
        // Skip if filter is set and names don't match
        if (filter) |f| {
            if (!std.mem.eql(u8, f, model.name)) continue;
        }

        // Build wasm target for this model
        buildWasmModel(
            b,
            model,
            wasm_target,
            wasm_tensor_module,
            wasm_ops_module,
            wasm_loader_module,
            wasm_step,
        );
    }
}

/// Build a WebAssembly target for a specific model
fn buildWasmModel(
    b: *std.Build,
    model: registry.ModelConfig,
    wasm_target: std.Build.ResolvedTarget,
    wasm_tensor_module: *std.Build.Module,
    wasm_ops_module: *std.Build.Module,
    wasm_loader_module: *std.Build.Module,
    wasm_step: *std.Build.Step,
) void {
    // Create the model module for Wasm
    const wasm_model_module = b.createModule(.{
        .root_source_file = b.path(model.model_module),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    // Map relative imports to our wasm modules
    wasm_model_module.addImport("../core/tensor.zig", wasm_tensor_module);
    wasm_model_module.addImport("../core/ops.zig", wasm_ops_module);
    wasm_model_module.addImport("../core/loader.zig", wasm_loader_module);

    // Create the binding module
    const wasm_binding = b.createModule(.{
        .root_source_file = b.path(model.wasm_binding),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    wasm_binding.addImport("tensor", wasm_tensor_module);
    wasm_binding.addImport("vad", wasm_model_module);

    // Embed model weights if bundled
    if (model.supports_bundled) {
        wasm_binding.addAnonymousImport("model_weights", .{
            .root_source_file = b.path(model.weights_path),
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

    // Also copy to example directory if specified
    if (model.example_dir) |example_dir| {
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
