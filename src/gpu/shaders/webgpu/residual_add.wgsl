// residual_add: In-place residual — A[i] += B[i]
//
// Dispatch: ((count+255)/256, 1, 1)

struct Params {
    count: u32,
};

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    a[idx] += b[idx];
}
