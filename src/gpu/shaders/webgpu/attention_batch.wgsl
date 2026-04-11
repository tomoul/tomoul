// attention_batch: Batched Multi-Head Self-Attention (packed sentences)
//
// One thread per (head, token). Determines sentence boundaries from offsets.
// Dispatch: (num_heads, total_tokens, 1), workgroup_size: (1, 1, 1)

struct Params {
    total_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
    batch_size: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<storage, read> offsets: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.x;
    let row = gid.y;
    if (head >= params.num_heads || row >= params.total_tokens) { return; }

    let hidden = params.num_heads * params.head_dim;
    let head_dim = params.head_dim;
    let scale = params.scale;

    // Find sentence boundaries
    var sent_start = 0u;
    var sent_end = params.total_tokens;
    for (var s = 0u; s < params.batch_size; s++) {
        let off = offsets[s];
        if (off <= row) {
            sent_start = off;
            if (s + 1u < params.batch_size) {
                sent_end = offsets[s + 1u];
            } else {
                sent_end = params.total_tokens;
            }
        }
    }
    let sent_len = sent_end - sent_start;
    let head_off = head * head_dim;

    // Step 1: scores + find max
    var max_score = -1e30f;
    for (var j = 0u; j < sent_len; j++) {
        let k_row = sent_start + j;
        var dot = 0.0f;
        for (var d = 0u; d < head_dim; d++) {
            dot += Q[row * hidden + head_off + d] * K[k_row * hidden + head_off + d];
        }
        max_score = max(max_score, dot * scale);
    }

    // Step 2: softmax
    var exp_sum = 0.0f;
    for (var j = 0u; j < sent_len; j++) {
        let k_row = sent_start + j;
        var dot = 0.0f;
        for (var d = 0u; d < head_dim; d++) {
            dot += Q[row * hidden + head_off + d] * K[k_row * hidden + head_off + d];
        }
        exp_sum += exp(dot * scale - max_score);
    }
    let inv_sum = 1.0 / (exp_sum + 1e-12);

    // Step 3: weighted sum of values
    for (var d = 0u; d < head_dim; d++) {
        var acc = 0.0f;
        for (var j = 0u; j < sent_len; j++) {
            let k_row = sent_start + j;
            var dot = 0.0f;
            for (var dd = 0u; dd < head_dim; dd++) {
                dot += Q[row * hidden + head_off + dd] * K[k_row * hidden + head_off + dd];
            }
            acc += exp(dot * scale - max_score) * inv_sum * V[k_row * hidden + head_off + d];
        }
        O[row * hidden + head_off + d] = acc;
    }
}
