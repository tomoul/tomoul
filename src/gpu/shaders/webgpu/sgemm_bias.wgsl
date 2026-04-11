// sgemm_bias: Tiled SGEMM with bias — C = A @ B + bias
//
// Workgroup: 16×16 threads, 64×64 output tile, 4×4 register blocking
// Dispatch: ((N+63)/64, (M+63)/64, 1)

struct Params {
    M: u32,
    N: u32,
    K: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const TILE: u32 = 64u;
const REG: u32 = 4u;

var<workgroup> As: array<array<f32, 64>, 64>;
var<workgroup> Bs: array<array<f32, 64>, 64>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let M = params.M;
    let N = params.N;
    let K = params.K;
    let row0 = wg.y * TILE + lid.y * REG;
    let col0 = wg.x * TILE + lid.x * REG;

    var acc: array<array<f32, 4>, 4>;
    for (var i = 0u; i < REG; i++) {
        for (var j = 0u; j < REG; j++) {
            acc[i][j] = 0.0;
        }
    }

    let num_tiles = (K + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t++) {
        let tile_k = t * TILE;

        // Load A tile
        for (var di = 0u; di < REG; di++) {
            for (var dj = 0u; dj < REG; dj++) {
                let sr = lid.y * REG + di;
                let sc = lid.x * REG + dj;
                let gr = wg.y * TILE + sr;
                let gc = tile_k + sc;
                if (gr < M && gc < K) {
                    As[sr][sc] = A[gr * K + gc];
                } else {
                    As[sr][sc] = 0.0;
                }
            }
        }

        // Load B tile
        for (var di = 0u; di < REG; di++) {
            for (var dj = 0u; dj < REG; dj++) {
                let sr = lid.y * REG + di;
                let sc = lid.x * REG + dj;
                let gr = tile_k + sr;
                let gc = wg.x * TILE + sc;
                if (gr < K && gc < N) {
                    Bs[sr][sc] = B[gr * N + gc];
                } else {
                    Bs[sr][sc] = 0.0;
                }
            }
        }

        workgroupBarrier();

        // Accumulate
        for (var k = 0u; k < TILE; k++) {
            for (var di = 0u; di < REG; di++) {
                let a_val = As[lid.y * REG + di][k];
                for (var dj = 0u; dj < REG; dj++) {
                    acc[di][dj] += a_val * Bs[k][lid.x * REG + dj];
                }
            }
        }

        workgroupBarrier();
    }

    // Write with bias
    for (var di = 0u; di < REG; di++) {
        for (var dj = 0u; dj < REG; dj++) {
            let r = row0 + di;
            let c = col0 + dj;
            if (r < M && c < N) {
                C[r * N + c] = acc[di][dj] + bias[c];
            }
        }
    }
}
