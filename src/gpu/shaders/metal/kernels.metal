// Metal Shading Language (MSL) compute kernels for Tomoul GPU backend
//
// Ports of the GLSL/Vulkan compute shaders to Metal.
// All kernels are compiled into a single MTLLibrary at runtime.
//
// Kernel list:
//   sgemm_bias         — Tiled SGEMM with bias: C = A @ B + bias
//   sgemm              — Tiled SGEMM with alpha/beta: C = alpha * A @ B + beta * C
//   layernorm          — In-place Layer Normalization
//   gelu               — Element-wise GELU activation
//   residual_add       — In-place residual: A[i] += B[i]
//   attention          — Fused Multi-Head Attention (single sentence)
//   attention_batch    — Batched Multi-Head Attention (packed sentences)
//   embedding_lookup   — Embedding lookup + sum (word + position + type)
//   pool_normalize     — Mean pooling + L2 normalization (single sentence)
//   pool_normalize_batch — Batched mean pooling + L2 normalization
//   vec_add            — Simple vector add: C = A + B (smoke test)

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Parameter Structs (passed via setBytes at buffer index after storage buffers)
// ============================================================================

struct SgemmBiasParams {
    uint M;
    uint N;
    uint K;
};

struct SgemmParams {
    uint M;
    uint N;
    uint K;
    float alpha;
    float beta;
};

struct LayerNormParams {
    uint rows;
    uint cols;
    float eps;
};

struct ElementParams {
    uint count;
};

struct AttentionParams {
    uint seq_len;
    uint num_heads;
    uint head_dim;
    float scale;
};

struct AttentionBatchParams {
    uint total_tokens;
    uint num_heads;
    uint head_dim;
    float scale;
    uint batch_size;
};

struct EmbeddingParams {
    uint seq_len;
    uint hidden_dim;
};

struct PoolNormParams {
    uint seq_len;
    uint hidden_dim;
};

struct PoolNormBatchParams {
    uint batch_size;
    uint hidden_dim;
};

struct VecAddParams {
    uint count;
};

// ============================================================================
// sgemm_bias: Tiled SGEMM with bias — C = A @ B + bias
//
// Workgroup: 16×16 threads, 64×64 output tile, 4×4 register blocking
// Dispatch: ((N+63)/64, (M+63)/64, 1)
// ============================================================================

constant uint TILE_SIZE = 64;
constant uint REG_M = 4;
constant uint REG_N = 4;

kernel void sgemm_bias(
    device const float* A          [[buffer(0)]],
    device const float* B          [[buffer(1)]],
    device float* C                [[buffer(2)]],
    device const float* bias       [[buffer(3)]],
    constant SgemmBiasParams& p    [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]]
) {
    uint M = p.M, N = p.N, K = p.K;
    uint row0 = tgid.y * TILE_SIZE + tid.y * REG_M;
    uint col0 = tgid.x * TILE_SIZE + tid.x * REG_N;

    float acc[REG_M][REG_N];
    for (uint i = 0; i < REG_M; i++)
        for (uint j = 0; j < REG_N; j++)
            acc[i][j] = 0.0f;

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint tile_k = t * TILE_SIZE;

        // Load tile of A into shared memory
        for (uint di = 0; di < REG_M; di++) {
            for (uint dj = 0; dj < REG_N; dj++) {
                uint sr = tid.y * REG_M + di;
                uint sc = tid.x * REG_N + dj;
                uint gr = tgid.y * TILE_SIZE + sr;
                uint gc = tile_k + sc;
                As[sr][sc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            }
        }

        // Load tile of B into shared memory
        for (uint di = 0; di < REG_M; di++) {
            for (uint dj = 0; dj < REG_N; dj++) {
                uint sr = tid.y * REG_M + di;
                uint sc = tid.x * REG_N + dj;
                uint gr = tile_k + sr;
                uint gc = tgid.x * TILE_SIZE + sc;
                Bs[sr][sc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate partial products
        for (uint k = 0; k < TILE_SIZE; k++) {
            for (uint di = 0; di < REG_M; di++) {
                float a_val = As[tid.y * REG_M + di][k];
                for (uint dj = 0; dj < REG_N; dj++) {
                    acc[di][dj] += a_val * Bs[k][tid.x * REG_N + dj];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results with bias
    for (uint di = 0; di < REG_M; di++) {
        for (uint dj = 0; dj < REG_N; dj++) {
            uint r = row0 + di;
            uint c = col0 + dj;
            if (r < M && c < N) {
                C[r * N + c] = acc[di][dj] + bias[c];
            }
        }
    }
}

// ============================================================================
// sgemm: Tiled SGEMM with alpha/beta — C = alpha * A @ B + beta * C
//
// Same tiling as sgemm_bias.
// Dispatch: ((N+63)/64, (M+63)/64, 1)
// ============================================================================

kernel void sgemm(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device float* C             [[buffer(2)]],
    constant SgemmParams& p     [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]]
) {
    uint M = p.M, N = p.N, K = p.K;
    float alpha = p.alpha, beta = p.beta;
    uint row0 = tgid.y * TILE_SIZE + tid.y * REG_M;
    uint col0 = tgid.x * TILE_SIZE + tid.x * REG_N;

    float acc[REG_M][REG_N];
    for (uint i = 0; i < REG_M; i++)
        for (uint j = 0; j < REG_N; j++)
            acc[i][j] = 0.0f;

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint tile_k = t * TILE_SIZE;

        for (uint di = 0; di < REG_M; di++) {
            for (uint dj = 0; dj < REG_N; dj++) {
                uint sr = tid.y * REG_M + di;
                uint sc = tid.x * REG_N + dj;
                uint gr = tgid.y * TILE_SIZE + sr;
                uint gc = tile_k + sc;
                As[sr][sc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            }
        }

        for (uint di = 0; di < REG_M; di++) {
            for (uint dj = 0; dj < REG_N; dj++) {
                uint sr = tid.y * REG_M + di;
                uint sc = tid.x * REG_N + dj;
                uint gr = tile_k + sr;
                uint gc = tgid.x * TILE_SIZE + sc;
                Bs[sr][sc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            for (uint di = 0; di < REG_M; di++) {
                float a_val = As[tid.y * REG_M + di][k];
                for (uint dj = 0; dj < REG_N; dj++) {
                    acc[di][dj] += a_val * Bs[k][tid.x * REG_N + dj];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint di = 0; di < REG_M; di++) {
        for (uint dj = 0; dj < REG_N; dj++) {
            uint r = row0 + di;
            uint c = col0 + dj;
            if (r < M && c < N) {
                uint idx = r * N + c;
                C[idx] = alpha * acc[di][dj] + beta * C[idx];
            }
        }
    }
}

// ============================================================================
// layernorm: In-place Layer Normalization
//
// Per-row: x = (x - mean) / sqrt(var + eps) * gamma + beta
// Workgroup: 256 threads (1D), one workgroup per row
// Uses parallel reduction for mean and variance
// Dispatch: (rows, 1, 1), threadsPerThreadgroup: (256, 1, 1)
// ============================================================================

kernel void layernorm(
    device float* data              [[buffer(0)]],
    device const float* gamma_buf   [[buffer(1)]],
    device const float* beta_buf    [[buffer(2)]],
    constant LayerNormParams& p     [[buffer(3)]],
    uint row   [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]]
) {
    uint cols = p.cols;
    float eps = p.eps;
    uint base = row * cols;

    threadgroup float sdata[256];

    // Phase 1: Compute mean (parallel reduction)
    float local_sum = 0.0f;
    for (uint i = tid; i < cols; i += 256) {
        local_sum += data[base + i];
    }
    sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sdata[0] / float(cols);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute variance
    float local_var = 0.0f;
    for (uint i = tid; i < cols; i += 256) {
        float diff = data[base + i] - mean;
        local_var += diff * diff;
    }
    sdata[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float variance = sdata[0] / float(cols);
    float inv_std = rsqrt(variance + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize in-place
    for (uint i = tid; i < cols; i += 256) {
        data[base + i] = (data[base + i] - mean) * inv_std * gamma_buf[i] + beta_buf[i];
    }
}

// ============================================================================
// gelu: Element-wise GELU activation (in-place)
//
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Uses Padé approximation for tanh
// Dispatch: ((count+255)/256, 1, 1), threadsPerThreadgroup: (256, 1, 1)
// ============================================================================

kernel void gelu(
    device float* data          [[buffer(0)]],
    constant ElementParams& p   [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= p.count) return;

    float x = data[idx];
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);

    // tanh via Padé approximation: tanh(t) ≈ t*(27+t²)/(27+9*t²), clamped
    float t = clamp(inner, -5.0f, 5.0f);
    float t2 = t * t;
    float tanh_val = t * (27.0f + t2) / (27.0f + 9.0f * t2);

    data[idx] = 0.5f * x * (1.0f + tanh_val);
}

// ============================================================================
// residual_add: In-place residual — A[i] += B[i]
//
// Dispatch: ((count+255)/256, 1, 1), threadsPerThreadgroup: (256, 1, 1)
// ============================================================================

kernel void residual_add(
    device float* a             [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    constant ElementParams& p   [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= p.count) return;
    a[idx] += b[idx];
}

// ============================================================================
// attention: Fused Multi-Head Self-Attention (single sentence)
//
// For each head: scores = Q_h @ K_h^T * scale, softmax, output = scores @ V_h
// Dispatch: (num_heads, seq_len, 1), threadsPerThreadgroup: (256, 1, 1)
// WorkGroupID.x = head index, WorkGroupID.y = output row
// ============================================================================

kernel void attention(
    device const float* q       [[buffer(0)]],
    device const float* k       [[buffer(1)]],
    device const float* v       [[buffer(2)]],
    device float* o             [[buffer(3)]],
    constant AttentionParams& p [[buffer(4)]],
    uint2 tgid    [[threadgroup_position_in_grid]],
    uint2 tid_vec [[thread_position_in_threadgroup]]
) {
    uint head = tgid.x;
    uint row = tgid.y;
    uint tid = tid_vec.x;
    uint seq_len = p.seq_len;
    uint head_dim = p.head_dim;
    float scale = p.scale;
    uint hidden = p.num_heads * head_dim;
    uint head_offset = head * head_dim;

    threadgroup float scores[512];   // max seq_len
    threadgroup float sdata[256];    // for parallel reductions

    // Phase 1: Compute attention scores
    for (uint j = tid; j < seq_len; j += 256) {
        float dp = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dp += q[row * hidden + head_offset + d] * k[j * hidden + head_offset + d];
        }
        scores[j] = dp * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Stable softmax — find max
    float local_max = -1e30f;
    for (uint j = tid; j < seq_len; j += 256) {
        local_max = max(local_max, scores[j]);
    }
    sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = sdata[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exp and sum
    float local_sum = 0.0f;
    for (uint j = tid; j < seq_len; j += 256) {
        scores[j] = exp(scores[j] - max_val);
        local_sum += scores[j];
    }
    sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_val = sdata[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize
    for (uint j = tid; j < seq_len; j += 256) {
        scores[j] /= sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Weighted sum
    for (uint d = tid; d < head_dim; d += 256) {
        float val = 0.0f;
        for (uint j = 0; j < seq_len; j++) {
            val += scores[j] * v[j * hidden + head_offset + d];
        }
        o[row * hidden + head_offset + d] = val;
    }
}

// ============================================================================
// attention_batch: Batched Multi-Head Self-Attention (packed sentences)
//
// Handles batched packed input where multiple sentences are concatenated.
// Each thread handles one (head, row) pair and determines sentence boundaries
// from the offsets buffer.
//
// Dispatch: (num_heads, total_tokens, 1), threadsPerThreadgroup: (1, 1, 1)
// ============================================================================

kernel void attention_batch(
    device const float* q_buf        [[buffer(0)]],
    device const float* k_buf        [[buffer(1)]],
    device const float* v_buf        [[buffer(2)]],
    device float* out_buf            [[buffer(3)]],
    device const uint* offsets       [[buffer(4)]],
    constant AttentionBatchParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint head = gid.x;
    uint row = gid.y;

    if (head >= p.num_heads || row >= p.total_tokens) return;

    uint hidden_dim = p.num_heads * p.head_dim;
    uint head_dim = p.head_dim;
    float scale = p.scale;

    // Find which sentence this row belongs to
    uint sent_start = 0;
    uint sent_end = p.total_tokens;
    for (uint s = 0; s < p.batch_size; s++) {
        uint off = offsets[s];
        if (off <= row) {
            sent_start = off;
            if (s + 1 < p.batch_size) {
                sent_end = offsets[s + 1];
            } else {
                sent_end = p.total_tokens;
            }
        }
    }
    uint sent_len = sent_end - sent_start;
    uint head_offset = head * head_dim;

    // Step 1: scaled dot-product scores, find max
    float max_score = -1e30f;
    for (uint j = 0; j < sent_len; j++) {
        uint k_row = sent_start + j;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q_buf[row * hidden_dim + head_offset + d]
                 * k_buf[k_row * hidden_dim + head_offset + d];
        }
        dot *= scale;
        max_score = max(max_score, dot);
    }

    // Step 2: softmax (exp and sum)
    float exp_sum = 0.0f;
    for (uint j = 0; j < sent_len; j++) {
        uint k_row = sent_start + j;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q_buf[row * hidden_dim + head_offset + d]
                 * k_buf[k_row * hidden_dim + head_offset + d];
        }
        exp_sum += exp(dot * scale - max_score);
    }
    float inv_sum = 1.0f / (exp_sum + 1e-12f);

    // Step 3: weighted sum of values
    for (uint d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (uint j = 0; j < sent_len; j++) {
            uint k_row = sent_start + j;
            float dot = 0.0f;
            for (uint dd = 0; dd < head_dim; dd++) {
                dot += q_buf[row * hidden_dim + head_offset + dd]
                     * k_buf[k_row * hidden_dim + head_offset + dd];
            }
            float attn = exp(dot * scale - max_score) * inv_sum;
            acc += attn * v_buf[k_row * hidden_dim + head_offset + d];
        }
        out_buf[row * hidden_dim + head_offset + d] = acc;
    }
}

// ============================================================================
// embedding_lookup: Embedding Lookup + Sum
//
// output[t, d] = word_emb[token_id[t], d] + pos_emb[pos_id[t], d] + type_emb[type_id[t], d]
// ids layout: [0..seq_len) = token_ids, [seq_len..2*seq_len) = pos_ids, [2*seq_len..3*seq_len) = type_ids
// Dispatch: ((seq_len*hidden_dim+255)/256, 1, 1), threadsPerThreadgroup: (256, 1, 1)
// ============================================================================

kernel void embedding_lookup(
    device const uint* ids          [[buffer(0)]],
    device const float* word_emb    [[buffer(1)]],
    device const float* pos_emb     [[buffer(2)]],
    device const float* type_emb    [[buffer(3)]],
    device float* output_buf        [[buffer(4)]],
    constant EmbeddingParams& p     [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    uint total = p.seq_len * p.hidden_dim;
    if (idx >= total) return;

    uint row = idx / p.hidden_dim;
    uint col = idx % p.hidden_dim;

    uint token_id = ids[row];
    uint pos_id = ids[p.seq_len + row];
    uint type_id = ids[2 * p.seq_len + row];

    output_buf[idx] = word_emb[token_id * p.hidden_dim + col]
                    + pos_emb[pos_id * p.hidden_dim + col]
                    + type_emb[type_id * p.hidden_dim + col];
}

// ============================================================================
// pool_normalize: Mean Pooling + L2 Normalization (single sentence)
//
// Phase 1: output[d] = mean(input[t][d]) for t in 0..seq_len-1
// Phase 2: output[d] /= L2_norm(output)
// Dispatch: (1, 1, 1), threadsPerThreadgroup: (256, 1, 1)
// ============================================================================

kernel void pool_normalize(
    device const float* input_buf   [[buffer(0)]],
    device float* output_buf        [[buffer(1)]],
    constant PoolNormParams& p      [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint seq_len = p.seq_len;
    uint hidden_dim = p.hidden_dim;

    threadgroup float pool_result[512];  // max hidden_dim
    threadgroup float sdata[256];

    // Phase 1: Mean pooling per dimension
    for (uint d = tid; d < hidden_dim; d += 256) {
        float sum = 0.0f;
        for (uint t = 0; t < seq_len; t++) {
            sum += input_buf[t * hidden_dim + d];
        }
        pool_result[d] = sum / float(seq_len);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute L2 norm
    float local_ss = 0.0f;
    for (uint d = tid; d < hidden_dim; d += 256) {
        local_ss += pool_result[d] * pool_result[d];
    }
    sdata[tid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float l2_norm = sqrt(sdata[0]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize and write output
    float inv_norm = (l2_norm > 1e-12f) ? (1.0f / l2_norm) : 0.0f;
    for (uint d = tid; d < hidden_dim; d += 256) {
        output_buf[d] = pool_result[d] * inv_norm;
    }
}

// ============================================================================
// pool_normalize_batch: Batched Mean Pooling + L2 Normalization
//
// One workgroup per sentence. Pools each sentence independently, then
// L2-normalizes each output embedding.
// Dispatch: (batch_size, 1, 1), threadsPerThreadgroup: (256, 1, 1)
// ============================================================================

kernel void pool_normalize_batch(
    device const float* input_buf       [[buffer(0)]],
    device float* output_buf            [[buffer(1)]],
    device const uint* offsets          [[buffer(2)]],
    device const uint* lengths          [[buffer(3)]],
    constant PoolNormBatchParams& p     [[buffer(4)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid       [[thread_position_in_threadgroup]]
) {
    if (batch_idx >= p.batch_size) return;

    uint hidden_dim = p.hidden_dim;
    uint offset = offsets[batch_idx];
    uint seq_len = lengths[batch_idx];

    threadgroup float pool_result[512];
    threadgroup float sdata[256];

    // Phase 1: Mean pooling per dimension
    for (uint d = tid; d < hidden_dim; d += 256) {
        float sum = 0.0f;
        for (uint t = 0; t < seq_len; t++) {
            sum += input_buf[(offset + t) * hidden_dim + d];
        }
        pool_result[d] = sum / float(seq_len);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute L2 norm
    float local_ss = 0.0f;
    for (uint d = tid; d < hidden_dim; d += 256) {
        local_ss += pool_result[d] * pool_result[d];
    }
    sdata[tid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float l2_norm = sqrt(sdata[0]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize and write output
    float inv_norm = (l2_norm > 1e-12f) ? (1.0f / l2_norm) : 0.0f;
    for (uint d = tid; d < hidden_dim; d += 256) {
        output_buf[batch_idx * hidden_dim + d] = pool_result[d] * inv_norm;
    }
}

// ============================================================================
// vec_add: Simple vector add — C[i] = A[i] + B[i]
//
// Smoke test for Metal compute dispatch.
// Dispatch: ((count+255)/256, 1, 1), threadsPerThreadgroup: (256, 1, 1)
// ============================================================================

kernel void vec_add(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device float* C             [[buffer(2)]],
    constant VecAddParams& p    [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < p.count) {
        C[idx] = A[idx] + B[idx];
    }
}
