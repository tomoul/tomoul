// pool_normalize_batch: Batched Mean Pooling + L2 Normalization
//
// One workgroup per sentence.
// Dispatch: (batch_size, 1, 1), workgroup_size: (256, 1, 1)

struct Params {
    batch_size: u32,
    hidden_dim: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> lengths: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> pool_result: array<f32, 512>;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let batch_idx = wg.x;
    if (batch_idx >= params.batch_size) { return; }

    let tid = lid.x;
    let hidden_dim = params.hidden_dim;
    let offset = offsets[batch_idx];
    let seq_len = lengths[batch_idx];

    // Phase 1: Mean pool
    for (var d = tid; d < hidden_dim; d += 256u) {
        var sum = 0.0f;
        for (var t = 0u; t < seq_len; t++) {
            sum += input[(offset + t) * hidden_dim + d];
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

    // Phase 3: Normalize
    for (var d = tid; d < hidden_dim; d += 256u) {
        output[batch_idx * hidden_dim + d] = pool_result[d] * inv_norm;
    }
}
