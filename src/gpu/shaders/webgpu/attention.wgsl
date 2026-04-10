// attention: Fused Multi-Head Self-Attention (single sentence)
//
// Dispatch: (num_heads, seq_len, 1), workgroup_size: (256, 1, 1)
// WG.x = head, WG.y = output row

struct Params {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> scores: array<f32, 512>;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let head = wg.x;
    let row = wg.y;
    let tid = lid.x;
    let seq_len = params.seq_len;
    let head_dim = params.head_dim;
    let scale = params.scale;
    let hidden = params.num_heads * head_dim;
    let head_off = head * head_dim;

    // Phase 1: Dot-product scores
    for (var j = tid; j < seq_len; j += 256u) {
        var dp = 0.0f;
        for (var d = 0u; d < head_dim; d++) {
            dp += Q[row * hidden + head_off + d] * K[j * hidden + head_off + d];
        }
        scores[j] = dp * scale;
    }
    workgroupBarrier();

    // Phase 2: Softmax — find max
    var local_max = -1e30f;
    for (var j = tid; j < seq_len; j += 256u) {
        local_max = max(local_max, scores[j]);
    }
    sdata[tid] = local_max;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        workgroupBarrier();
    }
    let max_val = sdata[0];
    workgroupBarrier();

    // Exp + sum
    var local_sum = 0.0f;
    for (var j = tid; j < seq_len; j += 256u) {
        scores[j] = exp(scores[j] - max_val);
        local_sum += scores[j];
    }
    sdata[tid] = local_sum;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        workgroupBarrier();
    }
    let sum_val = sdata[0];
    workgroupBarrier();

    // Normalize
    for (var j = tid; j < seq_len; j += 256u) {
        scores[j] /= sum_val;
    }
    workgroupBarrier();

    // Phase 3: Weighted sum
    for (var d = tid; d < head_dim; d += 256u) {
        var val = 0.0f;
        for (var j = 0u; j < seq_len; j++) {
            val += scores[j] * V[j * hidden + head_off + d];
        }
        O[row * hidden + head_off + d] = val;
    }
}
