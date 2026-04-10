// layernorm: In-place Layer Normalization
//
// Per-row: x = (x - mean) / sqrt(var + eps) * gamma + beta
// 256 threads per workgroup, parallel reduction
// Dispatch: (rows, 1, 1)

struct Params {
    rows: u32,
    cols: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wg.x;
    let tid = lid.x;
    let cols = params.cols;
    let eps = params.eps;
    let base = row * cols;

    // Phase 1: Mean
    var local_sum = 0.0f;
    for (var i = tid; i < cols; i += 256u) {
        local_sum += data[base + i];
    }
    sdata[tid] = local_sum;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        workgroupBarrier();
    }
    let mean = sdata[0] / f32(cols);
    workgroupBarrier();

    // Phase 2: Variance
    var local_var = 0.0f;
    for (var i = tid; i < cols; i += 256u) {
        let diff = data[base + i] - mean;
        local_var += diff * diff;
    }
    sdata[tid] = local_var;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        workgroupBarrier();
    }
    let inv_std = inverseSqrt(sdata[0] / f32(cols) + eps);
    workgroupBarrier();

    // Phase 3: Normalize
    for (var i = tid; i < cols; i += 256u) {
        data[base + i] = (data[base + i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
