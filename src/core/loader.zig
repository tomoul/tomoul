const std = @import("std");
const tensor_import = @import("tensor.zig");
const Tensor = tensor_import.Tensor;
const quantization = @import("quantization.zig");
const QuantizedTensorQ8 = quantization.QuantizedTensorQ8;

/// Error types for model loading operations
pub const LoadError = error{
    InvalidMagic,
    UnsupportedVersion,
    CorruptedFile,
    TensorNotFound,
    OutOfMemory,
    FileNotFound,
    InvalidShape,
    UnsupportedQuantFormat,
};

/// Quantization format codes
pub const QuantFormat = enum(u8) {
    f32 = 0, // Float32 (no quantization) - version 1
    q8_0 = 1, // Q8_0: symmetric 8-bit - version 1
    q4_0 = 2, // Q4_0: symmetric 4-bit (future)
};

/// Information about a tensor stored in the file
pub const TensorInfo = struct {
    name: []const u8,
    shape: []usize,
    data_offset: u64,
    data_size: u64,
};

/// Model loader for .tl binary files
/// Loads PyTorch weights exported by tools/export_basic.py or export_quantized.py
/// Version 1 supports both float32 and quantized formats (via quant_format byte)
pub const ModelLoader = struct {
    allocator: std.mem.Allocator,
    file_data: []const u8,
    owns_file_data: bool, // Track if we allocated file_data
    tensors: std.StringHashMap(TensorInfo),
    tensor_names: std.ArrayList([]const u8),
    version: u32,
    quant_format: QuantFormat, // Quantization format

    const Self = @This();
    const MAGIC = [4]u8{ 'T', 'O', 'U', 'L' };

    /// Load a .tl file from disk
    pub fn init(allocator: std.mem.Allocator, path: []const u8) !Self {
        // Open and read file
        const file = std.fs.cwd().openFile(path, .{}) catch |err| {
            return switch (err) {
                error.FileNotFound => LoadError.FileNotFound,
                else => LoadError.CorruptedFile,
            };
        };
        defer file.close();

        const file_size = try file.getEndPos();
        if (file_size < 16) {
            return LoadError.CorruptedFile;
        }

        const file_data = try allocator.alloc(u8, file_size);
        errdefer allocator.free(file_data);

        const bytes_read = try file.readAll(file_data);
        if (bytes_read != file_size) {
            return LoadError.CorruptedFile;
        }

        var loader = Self{
            .allocator = allocator,
            .file_data = file_data,
            .owns_file_data = true,
            .tensors = std.StringHashMap(TensorInfo).init(allocator),
            .tensor_names = .{},
            .version = 0,
            .quant_format = .f32,
        };

        try loader.parseHeader();
        return loader;
    }

    /// Load from embedded/pre-existing bytes (does not take ownership)
    /// Use this for Wasm builds where model is embedded via @embedFile
    pub fn initFromBytes(allocator: std.mem.Allocator, data: []const u8) !Self {
        if (data.len < 16) {
            return LoadError.CorruptedFile;
        }

        var loader = Self{
            .allocator = allocator,
            .file_data = data,
            .owns_file_data = false, // Don't free embedded data
            .tensors = std.StringHashMap(TensorInfo).init(allocator),
            .tensor_names = .{},
            .version = 0,
            .quant_format = .f32,
        };

        try loader.parseHeader();
        return loader;
    }

    /// Free all resources
    pub fn deinit(self: *Self) void {
        // Free tensor info (names and shapes)
        var iter = self.tensors.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.shape);
            self.allocator.free(entry.value_ptr.name);
        }
        self.tensors.deinit();

        // Free tensor names list
        self.tensor_names.deinit(self.allocator);

        // Free file data only if we allocated it
        if (self.owns_file_data) {
            self.allocator.free(self.file_data);
        }
    }

    /// Parse the file header and tensor table
    fn parseHeader(self: *Self) !void {
        if (self.file_data.len < 16) {
            return LoadError.CorruptedFile;
        }

        var pos: usize = 0;

        // Check magic bytes
        if (!std.mem.eql(u8, self.file_data[0..4], &MAGIC)) {
            return LoadError.InvalidMagic;
        }
        pos += 4;

        // Read version
        self.version = std.mem.readInt(u32, self.file_data[pos..][0..4], .little);
        pos += 4;

        // Support version 1 only
        if (self.version != 1) {
            return LoadError.UnsupportedVersion;
        }

        // Read tensor count
        const tensor_count = std.mem.readInt(u32, self.file_data[pos..][0..4], .little);
        pos += 4;

        // Read quantization format from reserved byte (first byte)
        const format_byte = self.file_data[pos];
        self.quant_format = std.meta.intToEnum(QuantFormat, format_byte) catch {
            return LoadError.UnsupportedQuantFormat;
        };
        pos += 4; // Skip full reserved field

        // Parse tensor table
        for (0..tensor_count) |_| {
            pos = try self.parseTensorEntry(pos);
        }
    }

    /// Parse a single tensor entry from the table
    fn parseTensorEntry(self: *Self, start_pos: usize) !usize {
        var pos = start_pos;

        // Read name length
        if (pos + 4 > self.file_data.len) return LoadError.CorruptedFile;
        const name_len = std.mem.readInt(u32, self.file_data[pos..][0..4], .little);
        pos += 4;

        // Read name
        if (pos + name_len > self.file_data.len) return LoadError.CorruptedFile;
        const name_slice = self.file_data[pos..][0..name_len];
        const name = try self.allocator.dupe(u8, name_slice);
        errdefer self.allocator.free(name);
        pos += name_len;

        // Read number of dimensions
        if (pos + 4 > self.file_data.len) return LoadError.CorruptedFile;
        const num_dims = std.mem.readInt(u32, self.file_data[pos..][0..4], .little);
        pos += 4;

        // Read shape
        if (pos + num_dims * 4 > self.file_data.len) return LoadError.CorruptedFile;
        const shape = try self.allocator.alloc(usize, num_dims);
        errdefer self.allocator.free(shape);

        for (shape) |*dim| {
            dim.* = std.mem.readInt(u32, self.file_data[pos..][0..4], .little);
            pos += 4;
        }

        // Read data offset and size
        if (pos + 16 > self.file_data.len) return LoadError.CorruptedFile;
        const data_offset = std.mem.readInt(u64, self.file_data[pos..][0..8], .little);
        pos += 8;
        const data_size = std.mem.readInt(u64, self.file_data[pos..][0..8], .little);
        pos += 8;

        // Store tensor info
        const info = TensorInfo{
            .name = name,
            .shape = shape,
            .data_offset = data_offset,
            .data_size = data_size,
        };

        try self.tensors.put(name, info);
        try self.tensor_names.append(self.allocator, name);

        return pos;
    }

    /// Get a tensor by name
    pub fn getTensor(self: *Self, name: []const u8) !Tensor {
        const info = self.tensors.get(name) orelse return LoadError.TensorNotFound;

        // Validate data bounds
        const end_offset = info.data_offset + info.data_size;
        if (end_offset > self.file_data.len) {
            return LoadError.CorruptedFile;
        }

        // Create tensor with the stored shape
        var tensor = try Tensor.init(self.allocator, info.shape);
        errdefer tensor.deinit();

        // Verify data size matches tensor size
        const expected_size = tensor.size() * @sizeOf(f32);
        if (info.data_size != expected_size) {
            return LoadError.CorruptedFile;
        }

        // Copy data from file buffer
        // Cast u64 to usize for slice indexing (safe on wasm32 since model files are small)
        const offset: usize = @intCast(info.data_offset);
        const size: usize = @intCast(info.data_size);
        const data_bytes = self.file_data[offset..][0..size];

        // Copy bytes to tensor data, handling potential unaligned data
        // Read f32 values byte-by-byte to handle unaligned memory
        for (tensor.data, 0..) |*out, i| {
            const byte_offset = i * 4;
            out.* = @bitCast([4]u8{
                data_bytes[byte_offset],
                data_bytes[byte_offset + 1],
                data_bytes[byte_offset + 2],
                data_bytes[byte_offset + 3],
            });
        }

        return tensor;
    }

    /// Get a quantized tensor by name (for Q8_0 format files)
    /// Returns the tensor in quantized form for weight-only inference
    pub fn getQuantizedTensorQ8(self: *Self, name: []const u8) !QuantizedTensorQ8 {
        if (self.quant_format != .q8_0) {
            return LoadError.UnsupportedQuantFormat;
        }

        const info = self.tensors.get(name) orelse return LoadError.TensorNotFound;

        // Validate data bounds
        const end_offset = info.data_offset + info.data_size;
        if (end_offset > self.file_data.len) {
            return LoadError.CorruptedFile;
        }

        // Create quantized tensor with the stored shape
        var qtensor = try QuantizedTensorQ8.init(self.allocator, info.shape);
        errdefer qtensor.deinit();

        // Q8_0 format: 4-byte scale + N bytes of int8 data
        const expected_size = 4 + qtensor.numel();
        if (info.data_size != expected_size) {
            return LoadError.CorruptedFile;
        }

        // Read scale (first 4 bytes)
        const offset: usize = @intCast(info.data_offset);
        qtensor.scale = @bitCast([4]u8{
            self.file_data[offset],
            self.file_data[offset + 1],
            self.file_data[offset + 2],
            self.file_data[offset + 3],
        });

        // Read int8 data (remaining bytes)
        const data_start = offset + 4;
        for (qtensor.data, 0..) |*out, i| {
            out.* = @bitCast(self.file_data[data_start + i]);
        }

        return qtensor;
    }

    /// Check if a tensor exists
    pub fn hasTensor(self: *const Self, name: []const u8) bool {
        return self.tensors.contains(name);
    }

    /// Check if this is a quantized model
    pub fn isQuantized(self: *const Self) bool {
        return self.quant_format != .f32;
    }

    /// Get the quantization format
    pub fn getQuantFormat(self: *const Self) QuantFormat {
        return self.quant_format;
    }

    /// Get number of tensors in the file
    pub fn tensorCount(self: *const Self) usize {
        return self.tensors.count();
    }

    /// Get list of tensor names
    pub fn listTensors(self: *const Self) []const []const u8 {
        return self.tensor_names.items;
    }

    /// Get tensor info without loading data
    pub fn getTensorInfo(self: *const Self, name: []const u8) ?TensorInfo {
        return self.tensors.get(name);
    }

    /// Print summary of loaded model (for debugging)
    pub fn printSummary(self: *const Self) void {
        std.debug.print("\n=== Model Summary ===\n", .{});
        std.debug.print("Version: {}\n", .{self.version});
        std.debug.print("Tensor count: {}\n", .{self.tensorCount()});
        std.debug.print("\nTensors:\n", .{});

        for (self.tensor_names.items) |name| {
            if (self.tensors.get(name)) |info| {
                std.debug.print("  {s}: shape=[", .{name});
                for (info.shape, 0..) |dim, i| {
                    if (i > 0) std.debug.print(", ", .{});
                    std.debug.print("{}", .{dim});
                }
                std.debug.print("], {} bytes\n", .{info.data_size});
            }
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "loader invalid magic" {
    const allocator = std.testing.allocator;

    // Create a fake file with wrong magic
    const bad_data = "BADM\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    const temp_path = "/tmp/tomoul_test_bad_magic.tl";

    // Write test file
    const file = try std.fs.cwd().createFile(temp_path, .{});
    try file.writeAll(bad_data);
    file.close();
    defer std.fs.cwd().deleteFile(temp_path) catch {};

    // Should fail with InvalidMagic
    const result = ModelLoader.init(allocator, temp_path);
    try std.testing.expectError(LoadError.InvalidMagic, result);
}

test "loader file not found" {
    const allocator = std.testing.allocator;

    const result = ModelLoader.init(allocator, "/nonexistent/path/model.tl");
    try std.testing.expectError(LoadError.FileNotFound, result);
}

test "load exported model" {
    const allocator = std.testing.allocator;

    // Load the linear model fixture (y = 2x + 1)
    var loader = ModelLoader.init(allocator, "tests/fixtures/linear/model.tl") catch |err| {
        // If file doesn't exist, skip test (run export_basic.py first)
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: fixtures not found. Run 'python3 tools/export_basic.py' first.\n", .{});
            return;
        }
        return err;
    };
    defer loader.deinit();

    // Verify tensor count (2 tensors: weight + bias for y = 2x + 1)
    try std.testing.expectEqual(@as(usize, 2), loader.tensorCount());

    // Load weight
    var weight = try loader.getTensor("weight");
    defer weight.deinit();

    // Verify shape: [1, 1] for nn.Linear(1, 1)
    try std.testing.expectEqual(@as(usize, 2), weight.shape.len);
    try std.testing.expectEqual(@as(usize, 1), weight.shape[0]);
    try std.testing.expectEqual(@as(usize, 1), weight.shape[1]);

    // Verify weight value (2.0 for y = 2x + 1)
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), weight.data[0], 0.0001);

    // Load bias
    var bias = try loader.getTensor("bias");
    defer bias.deinit();

    // Verify shape: [1]
    try std.testing.expectEqual(@as(usize, 1), bias.shape.len);
    try std.testing.expectEqual(@as(usize, 1), bias.shape[0]);

    // Verify bias value (1.0 for y = 2x + 1)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), bias.data[0], 0.0001);

    // Print summary for visual verification
    std.debug.print("\n", .{});
    loader.printSummary();
}

test "end-to-end linear layer forward pass" {
    const allocator = std.testing.allocator;

    // Load the model (y = 2x + 1) and validation data
    var model_loader = ModelLoader.init(allocator, "tests/fixtures/linear/model.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: fixtures not found.\n", .{});
            return;
        }
        return err;
    };
    defer model_loader.deinit();

    var val_loader = ModelLoader.init(allocator, "tests/fixtures/linear/validation.tl") catch |err| {
        if (err == LoadError.FileNotFound) {
            std.debug.print("\nSkipping test: validation fixtures not found.\n", .{});
            return;
        }
        return err;
    };
    defer val_loader.deinit();

    // Get model weights
    var weight = try model_loader.getTensor("weight");
    defer weight.deinit();
    var bias = try model_loader.getTensor("bias");
    defer bias.deinit();

    // Get validation input/output (x=10 -> y=21)
    var input = try val_loader.getTensor("input");
    defer input.deinit();
    var expected = try val_loader.getTensor("expected_output");
    defer expected.deinit();

    // Forward pass: y = x * weight + bias (for scalar: y = 10 * 2 + 1 = 21)
    const y_computed = input.data[0] * weight.data[0] + bias.data[0];

    // Verify result matches expected
    try std.testing.expectApproxEqAbs(expected.data[0], y_computed, 0.0001);

    std.debug.print("\nLinear layer forward pass result: ", .{});
    std.debug.print("x={d:.1} -> y={d:.1} (expected {d:.1})\n", .{ input.data[0], y_computed, expected.data[0] });
}
