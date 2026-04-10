// gelu: Element-wise GELU activation (in-place)
//
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Uses Padé tanh approximation
// Dispatch: ((count+255)/256, 1, 1)

struct Params {
    count: u32,
};

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }

    let x = data[idx];
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);

    // tanh via Padé: tanh(t) ≈ t*(27+t²)/(27+9*t²)
    let t = clamp(inner, -5.0, 5.0);
    let t2 = t * t;
    let tanh_val = t * (27.0 + t2) / (27.0 + 9.0 * t2);

    data[idx] = 0.5 * x * (1.0 + tanh_val);
}
