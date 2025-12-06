// src/core/blas.zig
// CBLAS interface for OpenBLAS integration
//
// This module provides extern declarations for CBLAS functions.
// When BLAS is enabled (-Dblas=true), these call into OpenBLAS.
// When disabled, this module is not imported at all.

const std = @import("std");

// CBLAS constants
pub const CBLAS_ORDER = enum(c_int) {
    RowMajor = 101,
    ColMajor = 102,
};

pub const CBLAS_TRANSPOSE = enum(c_int) {
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113,
};

// CBLAS extern declarations
// These are only resolved when linking against OpenBLAS
extern "c" fn cblas_sgemm(
    order: c_int,
    transA: c_int,
    transB: c_int,
    M: c_int,
    N: c_int,
    K: c_int,
    alpha: f32,
    A: [*]const f32,
    lda: c_int,
    B: [*]const f32,
    ldb: c_int,
    beta: f32,
    C: [*]f32,
    ldc: c_int,
) void;

extern "c" fn cblas_sgemv(
    order: c_int,
    trans: c_int,
    M: c_int,
    N: c_int,
    alpha: f32,
    A: [*]const f32,
    lda: c_int,
    X: [*]const f32,
    incX: c_int,
    beta: f32,
    Y: [*]f32,
    incY: c_int,
) void;

/// Single-precision General Matrix Multiply: C = alpha*A*B + beta*C
/// Wrapper around cblas_sgemm for row-major matrices.
///
/// Parameters:
///   - M: rows of A and C
///   - N: cols of B and C
///   - K: cols of A, rows of B
///   - A: matrix [M x K] in row-major order
///   - B: matrix [K x N] in row-major order
///   - C: output matrix [M x N] in row-major order
///   - alpha: scalar multiplier for A*B (typically 1.0)
///   - beta: scalar multiplier for C (0.0 to overwrite, 1.0 to accumulate)
pub fn sgemm(
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    B: []const f32,
    C: []f32,
    alpha: f32,
    beta: f32,
) void {
    cblas_sgemm(
        @intFromEnum(CBLAS_ORDER.RowMajor),
        @intFromEnum(CBLAS_TRANSPOSE.NoTrans),
        @intFromEnum(CBLAS_TRANSPOSE.NoTrans),
        @intCast(M),
        @intCast(N),
        @intCast(K),
        alpha,
        A.ptr,
        @intCast(K), // lda = K for row-major A[M,K]
        B.ptr,
        @intCast(N), // ldb = N for row-major B[K,N]
        beta,
        C.ptr,
        @intCast(N), // ldc = N for row-major C[M,N]
    );
}

/// Single-precision General Matrix Multiply with transpose options
/// C = alpha * op(A) * op(B) + beta * C
pub fn sgemmTranspose(
    transA: CBLAS_TRANSPOSE,
    transB: CBLAS_TRANSPOSE,
    M: usize,
    N: usize,
    K: usize,
    A: []const f32,
    lda: usize,
    B: []const f32,
    ldb: usize,
    C: []f32,
    ldc: usize,
    alpha: f32,
    beta: f32,
) void {
    cblas_sgemm(
        @intFromEnum(CBLAS_ORDER.RowMajor),
        @intFromEnum(transA),
        @intFromEnum(transB),
        @intCast(M),
        @intCast(N),
        @intCast(K),
        alpha,
        A.ptr,
        @intCast(lda),
        B.ptr,
        @intCast(ldb),
        beta,
        C.ptr,
        @intCast(ldc),
    );
}

/// Single-precision Matrix-Vector multiply: y = alpha*A*x + beta*y
pub fn sgemv(
    M: usize,
    N: usize,
    A: []const f32,
    x: []const f32,
    y: []f32,
    alpha: f32,
    beta: f32,
) void {
    cblas_sgemv(
        @intFromEnum(CBLAS_ORDER.RowMajor),
        @intFromEnum(CBLAS_TRANSPOSE.NoTrans),
        @intCast(M),
        @intCast(N),
        alpha,
        A.ptr,
        @intCast(N), // lda = N for row-major
        x.ptr,
        1, // incX
        beta,
        y.ptr,
        1, // incY
    );
}
