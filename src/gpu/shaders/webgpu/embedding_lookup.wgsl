// embedding_lookup: Embedding Lookup + Sum
//
// output[t,d] = word_emb[token_id[t],d] + pos_emb[pos_id[t],d] + type_emb[type_id[t],d]
// ids layout: [0..seq) = tokens, [seq..2*seq) = positions, [2*seq..3*seq) = types
// Dispatch: ((seq_len*hidden_dim+255)/256, 1, 1)

struct Params {
    seq_len: u32,
    hidden_dim: u32,
};

@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> word_emb: array<f32>;
@group(0) @binding(2) var<storage, read> pos_emb: array<f32>;
@group(0) @binding(3) var<storage, read> type_emb: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.hidden_dim;
    if (idx >= total) { return; }

    let row = idx / params.hidden_dim;
    let col = idx % params.hidden_dim;

    let token_id = ids[row];
    let pos_id = ids[params.seq_len + row];
    let type_id = ids[2u * params.seq_len + row];

    output[idx] = word_emb[token_id * params.hidden_dim + col]
                + pos_emb[pos_id * params.hidden_dim + col]
                + type_emb[type_id * params.hidden_dim + col];
}
