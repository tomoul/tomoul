// zblas/src/config.zig
// Compile-time configuration and tuning parameters for zblas
//
// These values control cache blocking and micro-kernel dimensions.
// Optimal values depend on CPU cache sizes and SIMD width.

const std = @import("std");
const builtin = @import("builtin");

/// Micro-kernel dimensions (MR x NR block computed per kernel call)
/// These must match the actual kernel implementations
pub const MR = switch (builtin.cpu.arch) {
    .x86_64 => 8, // AVX/AVX2: 8 floats per YMM register
    .aarch64 => 8, // NEON: 2x4 floats
    else => 4, // Generic: 4 floats per vector
};

pub const NR = switch (builtin.cpu.arch) {
    .x86_64 => 8,
    .aarch64 => 8,
    else => 4,
};

/// Cache blocking parameters (tuned for typical L2/L3 sizes)
/// MC: Rows of A block (fits in L2 with KC columns)
/// KC: K dimension block (shared dimension)
/// NC: Columns of B panel (fits in L3 with KC rows)
///
/// OpenBLAS Haswell parameters for reference:
///   MC (GEMM_P) = 320, KC (GEMM_Q) = 256, MR = 8, NR = 4
/// Key insight: KC=256 ensures packed A (MC×KC×4) fits in L2 cache
///   - MC=320, KC=256 => 320×256×4 = 320 KB (fits typical L2)
///   - MC=256, KC=512 => 256×512×4 = 512 KB (spills to L3!)
pub const MC = switch (builtin.cpu.arch) {
    .x86_64 => 320, // OpenBLAS value (was 256)
    .aarch64 => 256,
    else => 128,
};

pub const KC = switch (builtin.cpu.arch) {
    .x86_64 => 256, // OpenBLAS value (was 512) - critical for L2 fit!
    .aarch64 => 256, // Match OpenBLAS (was 512)
    else => 256,
};

pub const NC = switch (builtin.cpu.arch) {
    .x86_64 => 4096,
    .aarch64 => 4096,
    else => 1024,
};

/// Memory requirements for packing buffers
/// A buffer: MC * KC * sizeof(f32)
/// B buffer: KC * NC * sizeof(f32)
pub fn getPackedASize() usize {
    return MC * KC;
}

pub fn getPackedBSize() usize {
    return KC * NC;
}

/// Minimum matrix size to use optimized path (smaller uses reference)
pub const MIN_OPTIMIZED_SIZE = 32;

/// Alignment for packed buffers (cache line size)
pub const CACHE_LINE_SIZE = 64;

/// Check if architecture supports SIMD
pub fn hasSimd() bool {
    return switch (builtin.cpu.arch) {
        .x86_64 => true,
        .aarch64 => true,
        .wasm32 => true, // WASM SIMD
        else => false,
    };
}

/// Get the vector width in floats for current architecture
pub fn getVectorWidth() usize {
    return switch (builtin.cpu.arch) {
        .x86_64 => 8, // AVX: 256-bit = 8 floats
        .aarch64 => 4, // NEON: 128-bit = 4 floats
        .wasm32 => 4, // WASM SIMD: 128-bit = 4 floats
        else => 1, // Scalar fallback
    };
}

/// Tile sizes struct for compile-time configuration
pub const TileSizes = struct {
    mc: usize,
    kc: usize,
    nc: usize,
    mr: usize,
    nr: usize,
};

/// Get tile sizes for current architecture (compile-time)
pub fn getTileSizes() TileSizes {
    return .{
        .mc = MC,
        .kc = KC,
        .nc = NC,
        .mr = MR,
        .nr = NR,
    };
}

/// Cache configuration struct with all blocking parameters
/// Used for architecture-specific tuning
pub const CacheConfig = struct {
    mc: usize,
    kc: usize,
    nc: usize,
    mr: usize,
    nr: usize,

    /// Calculate packed A buffer size in bytes
    pub fn packedASize(self: CacheConfig) usize {
        return self.mc * self.kc * @sizeOf(f32);
    }

    /// Calculate packed B buffer size in bytes
    pub fn packedBSize(self: CacheConfig) usize {
        return self.kc * self.nc * @sizeOf(f32);
    }

    /// Check if packed A fits in L2 cache (256 KB typical)
    pub fn fitsL2(self: CacheConfig, l2_size: usize) bool {
        return self.packedASize() <= l2_size;
    }
};

/// Get optimal cache configuration for current architecture
/// Based on OpenBLAS analysis and cache size considerations
pub fn getCacheConfig() CacheConfig {
    const arch = builtin.cpu.arch;

    return switch (arch) {
        .x86_64 => .{
            // OpenBLAS Haswell-style parameters
            // Packed A = 320×256×4 = 320 KB (fits L2 with margin)
            .mc = 320,
            .kc = 256,
            .nc = 4096,
            .mr = 8,
            .nr = 8, // Keep 8 for now, test 4 separately
        },
        .aarch64 => .{
            // ARM64 optimized (Apple Silicon / Graviton)
            // Packed A = 256×256×4 = 256 KB (fits L2)
            .mc = 256,
            .kc = 256,
            .nc = 4096,
            .mr = 8,
            .nr = 8,
        },
        else => .{
            // Generic / fallback
            .mc = 128,
            .kc = 256,
            .nc = 2048,
            .mr = 4,
            .nr = 4,
        },
    };
}
