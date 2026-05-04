// Aggregator that pulls in every format/ test for `zig build test-format`.

comptime {
    _ = @import("dtype.zig");
    _ = @import("safetensors.zig");
    _ = @import("config_json.zig");
}
