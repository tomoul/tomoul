///! Debug program to print intermediate tensor values for comparison with PyTorch
///! This helps identify where Zig implementation diverges from PyTorch

const std = @import("std");
const model_mod = @import("model.zig");
const ops = @import("../../core/ops.zig");
const transformer = @import("../../core/transformer.zig");
const PunctuationModel = model_mod.PunctuationModel;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load model
    const weights_path = "artifacts/fullstop_punctuation_multilang_large.tl";
    const vocab_path = "artifacts/fullstop_punctuation_multilang_large_vocab.txt";

    std.debug.print("Loading model...\n", .{});
    var punct_model = try PunctuationModel.init(allocator, weights_path, vocab_path);
    defer punct_model.deinit();

    std.debug.print("Model loaded successfully!\n\n", .{});
    std.debug.print("{'='**60}\n", .{});
    std.debug.print("Testing with input: 'hello world'\n", .{});
    std.debug.print("{'='**60}\n\n", .{});

    // Test with hardcoded token IDs from PyTorch
    // PyTorch tokenizes "hello world" as: [0, 33600, 31, 8999, 2]
    // Tokens: ['<s>', '▁hell', 'o', '▁world', '</s>']
    const input_ids = [_]u32{ 0, 33600, 31, 8999, 2 };

    std.debug.print("1. INPUT TOKEN IDs: [", .{});
    for (input_ids, 0..) |id, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d}", .{id});
    }
    std.debug.print("]\n\n", .{});

    // Now let's manually run forward pass with debug prints
    try debugForward(&punct_model.model, &input_ids);
}

fn debugForward(model: *model_mod.XLMRobertaPunctuationModel, input_ids: []const u32) !void {
    const allocator = model.allocator;
    const seq_len = input_ids.len;
    const config = model.config;
    const transformer_config = config.getTransformerConfig();

    // 1. Embeddings: word + position + token_type
    std.debug.print("2. COMPUTING EMBEDDINGS...\n", .{});

    var word_emb = try ops.embedding(allocator, input_ids, &model.word_embeddings);
    defer word_emb.deinit();

    // Position IDs: 0, 1, 2, ...
    const position_ids = try allocator.alloc(u32, seq_len);
    defer allocator.free(position_ids);
    for (position_ids, 0..) |*p, i| {
        p.* = @intCast(i);
    }

    var pos_emb = try ops.embedding(allocator, position_ids, &model.position_embeddings);
    defer pos_emb.deinit();

    // Token type IDs: all zeros for single sequence
    const token_type_ids = try allocator.alloc(u32, seq_len);
    defer allocator.free(token_type_ids);
    @memset(token_type_ids, 0);

    var type_emb = try ops.embedding(allocator, token_type_ids, &model.token_type_embeddings);
    defer type_emb.deinit();

    // Combine embeddings
    try ops.addInPlace(&word_emb, &pos_emb);
    try ops.addInPlace(&word_emb, &type_emb);

    // Embedding layer norm
    var hidden = try ops.layerNorm(
        allocator,
        &word_emb,
        &model.embed_ln_gamma,
        &model.embed_ln_beta,
        config.layer_norm_eps,
    );
    defer hidden.deinit();

    std.debug.print("   Embeddings shape: [{d}, {d}]\n", .{ seq_len, config.hidden_size });
    std.debug.print("   First 5 values of token 0: [", .{});
    for (0..@min(5, config.hidden_size)) |i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d:.8}", .{hidden.data[i]});
    }
    std.debug.print("]\n", .{});

    std.debug.print("   First 5 values of token 1: [", .{});
    const offset = config.hidden_size;
    for (0..@min(5, config.hidden_size)) |i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d:.8}", .{hidden.data[offset + i]});
    }
    std.debug.print("]\n\n", .{});

    // 2. Run through transformer blocks (only first layer for debug)
    std.debug.print("3. LAYER 0 - TRANSFORMER BLOCK...\n", .{});

    const new_hidden = try transformer.transformerBlock(
        allocator,
        &hidden,
        &model.blocks[0],
        transformer_config,
    );
    hidden.deinit();
    hidden = new_hidden;

    std.debug.print("   After layer 0 shape: [{d}, {d}]\n", .{ seq_len, config.hidden_size });
    std.debug.print("   First 5 values of token 0: [", .{});
    for (0..@min(5, config.hidden_size)) |i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d:.8}", .{hidden.data[i]});
    }
    std.debug.print("]\n", .{});

    std.debug.print("   First 5 values of token 1: [", .{});
    for (0..@min(5, config.hidden_size)) |i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d:.8}", .{hidden.data[offset + i]});
    }
    std.debug.print("]\n\n", .{});

    // Run remaining layers
    std.debug.print("   Running remaining 23 layers...\n", .{});
    for (model.blocks[1..], 1..) |*block, layer_idx| {
        const layer_hidden = try transformer.transformerBlock(
            allocator,
            &hidden,
            block,
            transformer_config,
        );
        hidden.deinit();
        hidden = layer_hidden;

        if ((layer_idx + 1) % 6 == 0) {
            std.debug.print("   Completed layer {d}/24\n", .{layer_idx + 1});
        }
    }
    std.debug.print("\n", .{});

    // 3. Classification head
    std.debug.print("4. CLASSIFICATION HEAD...\n", .{});

    var logits = try ops.matmul(allocator, &hidden, &model.classifier_weight);
    defer logits.deinit();
    try ops.addBiasInPlace(&logits, &model.classifier_bias);

    const num_labels = 6;
    std.debug.print("   Logits shape: [{d}, {d}]\n", .{ seq_len, num_labels });
    std.debug.print("   Token 0 logits: [", .{});
    for (0..num_labels) |i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d:.8}", .{logits.data[i]});
    }
    std.debug.print("]\n", .{});

    std.debug.print("   Token 1 logits: [", .{});
    for (0..num_labels) |i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d:.8}", .{logits.data[num_labels + i]});
    }
    std.debug.print("]\n\n", .{});

    // 4. Argmax to get predictions
    std.debug.print("5. PREDICTIONS...\n", .{});
    std.debug.print("   Predicted label IDs: [", .{});

    for (0..seq_len) |token_idx| {
        if (token_idx > 0) std.debug.print(", ", .{});

        var max_val: f32 = -std.math.inf(f32);
        var max_idx: usize = 0;

        for (0..num_labels) |label_idx| {
            const val = logits.data[token_idx * num_labels + label_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = label_idx;
            }
        }

        std.debug.print("{d}", .{max_idx});
    }
    std.debug.print("]\n\n", .{});

    std.debug.print("{'='**60}\n", .{});
    std.debug.print("COMPARISON WITH PYTORCH\n", .{});
    std.debug.print("{'='**60}\n", .{});
    std.debug.print("Expected embeddings[0, 0, :5]: [-0.12444694, -0.23642007, -0.33971760, -0.29898580, 1.03427970]\n", .{});
    std.debug.print("Expected layer0[0, 0, :5]:      [-0.44686964, -0.38396580, -0.29896438, -0.53748480, 0.37506396]\n", .{});
    std.debug.print("Expected logits[0, 0, :]:       [-0.84641480, 3.28444960, -0.51838850, -4.05651500, -2.55231860, -3.63851570]\n", .{});
}
