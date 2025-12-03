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
    return b.fmt("models/{s}.tl", .{name});
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
    wasm_step: *std.Build.Step,
) void {
    // Derive paths from model name (convention over configuration)
    const model_module_path = deriveModelModule(b, model.name);
    const wasm_binding_path = deriveWasmBinding(b, model.name);
    const weights_path = deriveWeightsPath(b, model.name);

    // Create the model module for Wasm
    const wasm_model_module = b.createModule(.{
        .root_source_file = b.path(model_module_path),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    // Map relative imports to our wasm modules
    wasm_model_module.addImport("../../core/tensor.zig", wasm_tensor_module);
    wasm_model_module.addImport("../../core/ops.zig", wasm_ops_module);
    wasm_model_module.addImport("../../core/loader.zig", wasm_loader_module);

    // Create the binding module
    const wasm_binding = b.createModule(.{
        .root_source_file = b.path(wasm_binding_path),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
    });
    wasm_binding.addImport("tensor", wasm_tensor_module);
    wasm_binding.addImport("vad", wasm_model_module);

    // Embed model weights if bundled
    if (model.supports_bundled) {
        wasm_binding.addAnonymousImport("model_weights", .{
            .root_source_file = b.path(weights_path),
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
    // Derive paths from model name (convention over configuration)
    const c_binding_path = deriveCBinding(b, model.name);
    const model_module_path = deriveModelModule(b, model.name);
    const weights_path = deriveWeightsPath(b, model.name);

    // Create shared modules for native builds
    const tensor_module = b.createModule(.{
        .root_source_file = b.path("src/core/tensor.zig"),
        .target = target,
        .optimize = optimize,
    });

    const ops_module = b.createModule(.{
        .root_source_file = b.path("src/core/ops.zig"),
        .target = target,
        .optimize = optimize,
    });
    ops_module.addImport("tensor.zig", tensor_module);

    const loader_module = b.createModule(.{
        .root_source_file = b.path("src/core/loader.zig"),
        .target = target,
        .optimize = optimize,
    });
    loader_module.addImport("tensor.zig", tensor_module);

    // Create the model module
    const model_module = b.createModule(.{
        .root_source_file = b.path(model_module_path),
        .target = target,
        .optimize = optimize,
    });
    model_module.addImport("../../core/tensor.zig", tensor_module);
    model_module.addImport("../../core/ops.zig", ops_module);
    model_module.addImport("../../core/loader.zig", loader_module);

    // Create build options for bundled/lite mode
    const options = b.addOptions();
    options.addOption(bool, "embed_weights", bundled and model.supports_bundled);

    // Helper to configure a C binding module
    const configureBindingModule = struct {
        fn configure(
            builder: *std.Build,
            c_path: []const u8,
            tgt: std.Build.ResolvedTarget,
            opt: std.builtin.OptimizeMode,
            tensor_mod: *std.Build.Module,
            model_mod: *std.Build.Module,
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
        options,
        bundled,
        model.supports_bundled,
        weights_path,
    );

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

    // Note: C header generation is handled by release.py
    // Zig's -femit-h doesn't work reliably with complex module dependencies

    // Create C binding module for shared library (modules can only be used once)
    const c_binding_module_shared = configureBindingModule(
        b,
        c_binding_path,
        target,
        optimize,
        tensor_module,
        model_module,
        options,
        bundled,
        model.supports_bundled,
        weights_path,
    );

    const shared_lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = lib_name,
        .root_module = c_binding_module_shared,
    });

    const shared_install = b.addInstallArtifact(shared_lib, .{});
    lib_step.dependOn(&shared_install.step);
}
