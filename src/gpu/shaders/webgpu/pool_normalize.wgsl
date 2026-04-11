// pool_normalize: Mean Pooling + L2 Normalization (single sentence)
//
// Phase 1: mean pool across tokens
// Phase 2: L2 normalize
// Dispatch: (1, 1, 1), workgroup_size: (256, 1, 1)

struct Params {
    seq_len: u32,
    hidden_dim: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> pool_result: array<f32, 512>;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let seq_len = params.seq_len;
    let hidden_dim = params.hidden_dim;

    // Phase 1: Mean pool
    for (var d = tid; d < hidden_dim; d += 256u) {
        var sum = 0.0f;
        for (var t = 0u; t < seq_len; t++) {
            sum += input[t * hidden_dim + d];
        }
        pool_result[d] = sum / f32(seq_len);
    }
    workgroupBarrier();

    // Phase 2: L2 norm
    var local_ss = 0.0f;
    for (var d = tid; d < hidden_dim; d += 256u) {
        local_ss += pool_result[d] * pool_result[d];
    }
    sdata[tid] = local_ss;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        workgroupBarrier();
    }
    let inv_norm = inverseSqrt(max(sdata[0], 1e-24));
    workgroupBarrier();

    // Phase 3: Normalize output
    for (var d = tid; d < hidden_dim; d += 256u) {
        output[d] = pool_result[d] * inv_norm;
    }
}
