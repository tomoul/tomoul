///! Integration Tests for Tomoul
///!
///! These tests use fixtures from tests/fixtures/ that can be committed to git.
///! They do not require downloading large models from the internet.
///!
///! Run with: zig build test-integration
///!
const std = @import("std");
const tomoul = @import("tomoul");
const Tensor = tomoul.Tensor;
const ops = tomoul.ops;
const ModelLoader = tomoul.loader.ModelLoader;
const LoadError = tomoul.loader.LoadError;

// ============================================================================
// Linear Model Tests (y = 2x + 1)
// ============================================================================

test "Integration: Linear model y=2x+1 parity" {
    const allocator = std.testing.allocator;

    // Load model weights
    var model_loader = ModelLoader.init(allocator, "tests/fixtures/linear/model.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: fixtures not found. Run 'python3 tools/export_basic.py' first.\n", .{});
            return;
        }
        return err;
    };
    defer model_loader.deinit();

    // Load validation data
    var val_loader = ModelLoader.init(allocator, "tests/fixtures/linear/validation.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: validation fixtures not found.\n", .{});
            return;
        }
        return err;
    };
    defer val_loader.deinit();

    // Get model weights (y = 2x + 1)
    var weight = try model_loader.getTensor("weight");
    defer weight.deinit();
    var bias = try model_loader.getTensor("bias");
    defer bias.deinit();

    // Verify weights are correct
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), weight.data[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), bias.data[0], 0.0001);

    // Test case 1: input = 10.0, expected = 21.0
    var input = try val_loader.getTensor("input");
    defer input.deinit();
    var expected = try val_loader.getTensor("expected_output");
    defer expected.deinit();

    // Compute: y = x * weight + bias
    const x_val = input.data[0];
    const y_computed = x_val * weight.data[0] + bias.data[0];

    std.debug.print("\nLinear model test: x={d:.1} -> y={d:.1} (expected {d:.1})\n", .{ x_val, y_computed, expected.data[0] });
    try std.testing.expectApproxEqAbs(expected.data[0], y_computed, 0.0001);

    // Test case 2: input = 0.0, expected = 1.0
    var input_zero = try val_loader.getTensor("input_zero");
    defer input_zero.deinit();
    var expected_zero = try val_loader.getTensor("expected_output_zero");
    defer expected_zero.deinit();

    const y_zero = input_zero.data[0] * weight.data[0] + bias.data[0];
    std.debug.print("Linear model test: x={d:.1} -> y={d:.1} (expected {d:.1})\n", .{ input_zero.data[0], y_zero, expected_zero.data[0] });
    try std.testing.expectApproxEqAbs(expected_zero.data[0], y_zero, 0.0001);

    // Test case 3: input = -5.0, expected = -9.0
    var input_neg = try val_loader.getTensor("input_negative");
    defer input_neg.deinit();
    var expected_neg = try val_loader.getTensor("expected_output_negative");
    defer expected_neg.deinit();

    const y_neg = input_neg.data[0] * weight.data[0] + bias.data[0];
    std.debug.print("Linear model test: x={d:.1} -> y={d:.1} (expected {d:.1})\n", .{ input_neg.data[0], y_neg, expected_neg.data[0] });
    try std.testing.expectApproxEqAbs(expected_neg.data[0], y_neg, 0.0001);
}

// ============================================================================
// Tiny VAD Model Tests
// ============================================================================

test "Integration: Tiny VAD model loads correctly" {
    const allocator = std.testing.allocator;

    // Load tiny VAD model
    var loader = ModelLoader.init(allocator, "tests/fixtures/silero_vad/model_tiny.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: fixtures not found. Run 'python3 tools/export_vad.py --tiny' first.\n", .{});
            return;
        }
        return err;
    };
    defer loader.deinit();

    // Verify tensor count
    const tensor_count = loader.tensorCount();
    std.debug.print("\nTiny VAD: loaded {} tensors\n", .{tensor_count});
    try std.testing.expectEqual(@as(usize, 15), tensor_count);

    // Verify key tensors exist
    try std.testing.expect(loader.hasTensor("_model.stft.forward_basis_buffer"));
    try std.testing.expect(loader.hasTensor("_model.decoder.rnn.weight_ih"));
    try std.testing.expect(loader.hasTensor("_model.decoder.decoder.2.weight"));

    // Load and verify STFT basis shape
    var stft = try loader.getTensor("_model.stft.forward_basis_buffer");
    defer stft.deinit();
    try std.testing.expectEqual(@as(usize, 3), stft.shape.len);
    try std.testing.expectEqual(@as(usize, 32), stft.shape[0]); // Tiny: 16*2

    // Load and verify LSTM weights shape
    var lstm_ih = try loader.getTensor("_model.decoder.rnn.weight_ih");
    defer lstm_ih.deinit();
    try std.testing.expectEqual(@as(usize, 2), lstm_ih.shape.len);
    try std.testing.expectEqual(@as(usize, 32), lstm_ih.shape[0]); // 4 * hidden_size=8
    try std.testing.expectEqual(@as(usize, 8), lstm_ih.shape[1]); // hidden_size=8

    std.debug.print("Tiny VAD model structure verified.\n", .{});
}

test "Integration: Tiny VAD validation data loads" {
    const allocator = std.testing.allocator;

    // Load validation data
    var loader = ModelLoader.init(allocator, "tests/fixtures/silero_vad/validation.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: fixtures not found.\n", .{});
            return;
        }
        return err;
    };
    defer loader.deinit();

    // Verify validation tensors exist
    try std.testing.expect(loader.hasTensor("input_zeros"));
    try std.testing.expect(loader.hasTensor("output_zeros"));
    try std.testing.expect(loader.hasTensor("input_ones"));
    try std.testing.expect(loader.hasTensor("output_ones"));

    // Load and verify shapes
    var input_zeros = try loader.getTensor("input_zeros");
    defer input_zeros.deinit();
    try std.testing.expectEqual(@as(usize, 2), input_zeros.shape.len);
    try std.testing.expectEqual(@as(usize, 1), input_zeros.shape[0]); // batch=1
    try std.testing.expectEqual(@as(usize, 64), input_zeros.shape[1]); // Tiny input size

    var output_zeros = try loader.getTensor("output_zeros");
    defer output_zeros.deinit();
    // Output should be a probability between 0 and 1
    try std.testing.expect(output_zeros.data[0] >= 0.0);
    try std.testing.expect(output_zeros.data[0] <= 1.0);

    std.debug.print("\nValidation data loaded. Output for zeros: {d:.6}\n", .{output_zeros.data[0]});
}

// ============================================================================
// LSTM Operation Tests
// ============================================================================

test "Integration: LSTM with tiny VAD weights" {
    const allocator = std.testing.allocator;

    // Load tiny VAD model
    var loader = ModelLoader.init(allocator, "tests/fixtures/silero_vad/model_tiny.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: fixtures not found.\n", .{});
            return;
        }
        return err;
    };
    defer loader.deinit();

    // Load LSTM weights
    var weight_ih = try loader.getTensor("_model.decoder.rnn.weight_ih");
    defer weight_ih.deinit();
    var weight_hh = try loader.getTensor("_model.decoder.rnn.weight_hh");
    defer weight_hh.deinit();
    var bias_ih = try loader.getTensor("_model.decoder.rnn.bias_ih");
    defer bias_ih.deinit();
    var bias_hh = try loader.getTensor("_model.decoder.rnn.bias_hh");
    defer bias_hh.deinit();

    const hidden_size: usize = 8; // Tiny model hidden size

    // Create LSTM state
    var state = try ops.LSTMState.init(allocator, hidden_size);
    defer state.deinit();

    // Create input tensor (matching hidden_size for simplicity)
    var input_shape = [_]usize{hidden_size};
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    input.fill(0.1);

    // Create weights struct
    const weights = ops.LSTMWeights{
        .weight_ih = weight_ih,
        .weight_hh = weight_hh,
        .bias_ih = bias_ih,
        .bias_hh = bias_hh,
    };

    // Run LSTM cell
    try ops.lstmCell(allocator, &input, &state, &weights);

    // Verify hidden state is non-zero
    var h_sum: f32 = 0;
    for (state.h.data) |v| h_sum += @abs(v);
    try std.testing.expect(h_sum > 0.0);

    std.debug.print("\nLSTM forward pass: h_sum={d:.6}\n", .{h_sum});

    // Run another step and verify state changes
    const old_h0 = state.h.data[0];
    try ops.lstmCell(allocator, &input, &state, &weights);
    try std.testing.expect(state.h.data[0] != old_h0);

    std.debug.print("LSTM state evolves correctly over multiple steps.\n", .{});
}

// ============================================================================
// Conv1D Operation Tests
// ============================================================================

test "Integration: Conv1D with tiny VAD encoder weights" {
    const allocator = std.testing.allocator;

    // Load tiny VAD model
    var loader = ModelLoader.init(allocator, "tests/fixtures/silero_vad/model_tiny.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: fixtures not found.\n", .{});
            return;
        }
        return err;
    };
    defer loader.deinit();

    // Load first encoder layer weights
    var enc0_weight = try loader.getTensor("_model.encoder.0.reparam_conv.weight");
    defer enc0_weight.deinit();
    var enc0_bias = try loader.getTensor("_model.encoder.0.reparam_conv.bias");
    defer enc0_bias.deinit();

    // Tiny model: [8, 17, 3] (out_channels=8, in_channels=17, kernel=3)
    try std.testing.expectEqual(@as(usize, 3), enc0_weight.shape.len);
    const out_channels = enc0_weight.shape[0];
    const in_channels = enc0_weight.shape[1];
    const kernel_size = enc0_weight.shape[2];

    std.debug.print("\nEncoder layer 0: [{}, {}, {}]\n", .{ out_channels, in_channels, kernel_size });

    // Create a small input tensor matching in_channels
    const input_width: usize = 10;
    var input_shape = [_]usize{ in_channels, input_width };
    var input = try Tensor.init(allocator, &input_shape);
    defer input.deinit();
    input.fill(0.1);

    // Run Conv1D
    var output = try ops.conv1d(allocator, &input, &enc0_weight, &enc0_bias, 1, 1);
    defer output.deinit();

    // Verify output shape
    try std.testing.expectEqual(@as(usize, 2), output.shape.len);
    try std.testing.expectEqual(out_channels, output.shape[0]);

    std.debug.print("Conv1D output shape: [{}, {}]\n", .{ output.shape[0], output.shape[1] });
}
