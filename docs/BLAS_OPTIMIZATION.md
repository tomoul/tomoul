# BLAS Optimization for Tomoul

- `build.zig` - Added `-Dblas` option with conditional OpenBLAS linking
- `src/core/blas.zig` - CBLAS extern declarations (`cblas_sgemm`, `cblas_sgemv`)
- `src/core/ops.zig` - Conditional dispatch: uses BLAS when enabled, pure Zig otherwise

