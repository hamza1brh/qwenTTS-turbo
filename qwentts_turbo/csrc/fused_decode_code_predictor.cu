/**
 * Fused Decode for Qwen3-TTS Code Predictor
 *
 * Key differences from backbone (fused_decode_tts.cu):
 * - Uses STANDARD 1D RoPE (not multimodal interleaved)
 * - 5 layers instead of 28
 * - Same dimensions otherwise (1024 hidden, 16 Q heads, 8 KV heads, 128 head_dim)
 */

#include "config.cuh"  // Use same config as backbone - same dimensions
#include <cooperative_groups.h>

// Code predictor constants (using same values from config.cuh, just aliased for clarity)
constexpr int CP_HIDDEN_SIZE = HIDDEN_SIZE;           // 1024
constexpr int CP_INTERMEDIATE_SIZE = INTERMEDIATE_SIZE; // 3072
constexpr int CP_NUM_Q_HEADS = NUM_Q_HEADS;           // 16
constexpr int CP_NUM_KV_HEADS = NUM_KV_HEADS;         // 8
constexpr int CP_HEAD_DIM = HEAD_DIM;                 // 128
constexpr int CP_Q_SIZE = Q_SIZE;                     // 2048
constexpr int CP_KV_SIZE = KV_SIZE;                   // 1024
constexpr float CP_RMS_NORM_EPS = RMS_NORM_EPS;       // 1e-6

namespace cg = cooperative_groups;

// Kernel configuration
constexpr int CP_NUM_BLOCKS = 82;
constexpr int CP_BLOCK_SIZE = 256;
constexpr int CP_NUM_WARPS_KERNEL = CP_BLOCK_SIZE / WARP_SIZE;

struct CPLayerWeights {
    const __nv_bfloat16* input_layernorm_weight;
    const __nv_bfloat16* q_proj_weight;
    const __nv_bfloat16* k_proj_weight;
    const __nv_bfloat16* v_proj_weight;
    const __nv_bfloat16* q_norm_weight;
    const __nv_bfloat16* k_norm_weight;
    const __nv_bfloat16* o_proj_weight;
    const __nv_bfloat16* post_attn_layernorm_weight;
    const __nv_bfloat16* gate_proj_weight;
    const __nv_bfloat16* up_proj_weight;
    const __nv_bfloat16* down_proj_weight;
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float cp_warp_reduce_sum_kernel(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float cp_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// QKV Projection with RMSNorm
// =============================================================================

__device__ void cp_matvec_qkv(
    cg::grid_group& grid,
    const float* __restrict__ input,  // fp32 hidden buffer
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ v_weight,
    float* __restrict__ g_normalized,
    float* __restrict__ g_residual,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Block 0 does RMSNorm
    if (block_id == 0) {
        __shared__ float smem[CP_HIDDEN_SIZE];
        __shared__ float smem_reduce[CP_NUM_WARPS_KERNEL];

        float local_sum_sq = 0.0f;

        for (int i = threadIdx.x; i < CP_HIDDEN_SIZE; i += CP_BLOCK_SIZE) {
            float v = __ldg(input + i);  // fp32 read
            smem[i] = v;
            g_residual[i] = v;
            local_sum_sq += v * v;
        }

        local_sum_sq = cp_warp_reduce_sum_kernel(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < CP_NUM_WARPS_KERNEL) ? smem_reduce[lane_id] : 0.0f;
            sum = cp_warp_reduce_sum_kernel(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(CP_HIDDEN_SIZE) + CP_RMS_NORM_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x; i < CP_HIDDEN_SIZE; i += CP_BLOCK_SIZE) {
            float w = __bfloat162float(__ldg(norm_weight + i));
            g_normalized[i] = smem[i] * rstd * w;
        }
    }

    grid.sync();

    // QKV projection
    constexpr int TOTAL_ROWS = CP_Q_SIZE + CP_KV_SIZE + CP_KV_SIZE;
    int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

    for (int m_base = row_start; m_base < row_end; m_base += CP_NUM_WARPS_KERNEL) {
        int m = m_base + warp_id;

        if (m < row_end) {
            const __nv_bfloat16* weight_row;
            float* output_ptr;

            if (m < CP_Q_SIZE) {
                weight_row = q_weight + m * CP_HIDDEN_SIZE;
                output_ptr = q_out + m;
            } else if (m < CP_Q_SIZE + CP_KV_SIZE) {
                weight_row = k_weight + (m - CP_Q_SIZE) * CP_HIDDEN_SIZE;
                output_ptr = k_out + (m - CP_Q_SIZE);
            } else {
                weight_row = v_weight + (m - CP_Q_SIZE - CP_KV_SIZE) * CP_HIDDEN_SIZE;
                output_ptr = v_out + (m - CP_Q_SIZE - CP_KV_SIZE);
            }

            float sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < CP_HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(weight_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum += __bfloat162float(w_ptr[0]) * g_normalized[k] +
                       __bfloat162float(w_ptr[1]) * g_normalized[k+1] +
                       __bfloat162float(w_ptr[2]) * g_normalized[k+2] +
                       __bfloat162float(w_ptr[3]) * g_normalized[k+3];
            }

            sum = cp_warp_reduce_sum_kernel(sum);
            if (lane_id == 0) {
                *output_ptr = sum;
            }
        }
    }

    grid.sync();
}

// =============================================================================
// QK Norm + STANDARD 1D RoPE + KV Cache
// =============================================================================

__device__ void cp_qk_norm_rope_cache(
    cg::grid_group& grid,
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,  // [max_seq_len, head_dim] STANDARD format
    const __nv_bfloat16* __restrict__ sin_table,  // [max_seq_len, head_dim] STANDARD format
    float* __restrict__ k_cache,   // fp32 for precision with short context
    float* __restrict__ v_cache,   // fp32 for precision with short context
    int position,
    int max_seq_len
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Standard 1D RoPE: cos/sin are [head_dim] for each position (PyTorch format)
    const __nv_bfloat16* cos_pos = cos_table + position * CP_HEAD_DIM;
    const __nv_bfloat16* sin_pos = sin_table + position * CP_HEAD_DIM;

    // Process Q heads
    int q_heads_per_block = (CP_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int q_head_start = block_id * q_heads_per_block;
    int q_head_end = min(q_head_start + q_heads_per_block, CP_NUM_Q_HEADS);

    for (int h = q_head_start + warp_id; h < q_head_end; h += CP_NUM_WARPS_KERNEL) {
        float* q_head = q + h * CP_HEAD_DIM;

        // RMSNorm on Q head
        float sum_sq = 0.0f;
        for (int i = lane_id; i < CP_HEAD_DIM; i += WARP_SIZE) {
            sum_sq += q_head[i] * q_head[i];
        }
        sum_sq = cp_warp_reduce_sum_kernel(sum_sq);
        float scale = rsqrtf(sum_sq / float(CP_HEAD_DIM) + CP_RMS_NORM_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        // Apply norm and store locally
        float q_local[CP_HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < CP_HEAD_DIM; i += WARP_SIZE, j++) {
            q_local[j] = q_head[i] * scale * __bfloat162float(__ldg(q_norm_weight + i));
        }

        // PyTorch-style RoPE: q_embed = q * cos + rotate_half(q) * sin
        // rotate_half(q)[i] = -q[i+half] if i < half, else q[i-half]
        #pragma unroll
        for (int i = lane_id, j = 0; i < CP_HEAD_DIM; i += WARP_SIZE, j++) {
            int half_dim = CP_HEAD_DIM / 2;
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));

            float rot_half;
            if (i < half_dim) {
                // rotate_half[i] = -x[i+half]
                int pair_idx = i + half_dim;
                int pair_j = pair_idx / WARP_SIZE;
                rot_half = -__shfl_sync(0xffffffff, q_local[pair_j], pair_idx % WARP_SIZE);
            } else {
                // rotate_half[i] = x[i-half]
                int pair_idx = i - half_dim;
                int pair_j = pair_idx / WARP_SIZE;
                rot_half = __shfl_sync(0xffffffff, q_local[pair_j], pair_idx % WARP_SIZE);
            }
            q_head[i] = q_local[j] * cos_v + rot_half * sin_v;
        }
    }

    // Process K heads + cache
    int k_heads_per_block = (CP_NUM_KV_HEADS + num_blocks - 1) / num_blocks;
    int k_head_start = block_id * k_heads_per_block;
    int k_head_end = min(k_head_start + k_heads_per_block, CP_NUM_KV_HEADS);

    for (int h = k_head_start + warp_id; h < k_head_end; h += CP_NUM_WARPS_KERNEL) {
        float* k_head = k + h * CP_HEAD_DIM;
        const float* v_head = v + h * CP_HEAD_DIM;
        float* k_cache_head = k_cache + h * max_seq_len * CP_HEAD_DIM + position * CP_HEAD_DIM;
        float* v_cache_head = v_cache + h * max_seq_len * CP_HEAD_DIM + position * CP_HEAD_DIM;

        // RMSNorm on K head
        float sum_sq = 0.0f;
        for (int i = lane_id; i < CP_HEAD_DIM; i += WARP_SIZE) {
            sum_sq += k_head[i] * k_head[i];
        }
        sum_sq = cp_warp_reduce_sum_kernel(sum_sq);
        float scale = rsqrtf(sum_sq / float(CP_HEAD_DIM) + CP_RMS_NORM_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float k_local[CP_HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < CP_HEAD_DIM; i += WARP_SIZE, j++) {
            k_local[j] = k_head[i] * scale * __bfloat162float(__ldg(k_norm_weight + i));
        }

        // PyTorch-style RoPE for K + write to cache
        #pragma unroll
        for (int i = lane_id, j = 0; i < CP_HEAD_DIM; i += WARP_SIZE, j++) {
            int half_dim = CP_HEAD_DIM / 2;
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));

            float rot_half;
            if (i < half_dim) {
                int pair_idx = i + half_dim;
                int pair_j = pair_idx / WARP_SIZE;
                rot_half = -__shfl_sync(0xffffffff, k_local[pair_j], pair_idx % WARP_SIZE);
            } else {
                int pair_idx = i - half_dim;
                int pair_j = pair_idx / WARP_SIZE;
                rot_half = __shfl_sync(0xffffffff, k_local[pair_j], pair_idx % WARP_SIZE);
            }
            float k_final = k_local[j] * cos_v + rot_half * sin_v;

            k_head[i] = k_final;
            k_cache_head[i] = k_final;      // fp32 cache for precision
            v_cache_head[i] = v_head[i];    // fp32 cache for precision
        }
    }

    grid.sync();
}

// =============================================================================
// Attention
// =============================================================================

__device__ void cp_attention(
    cg::grid_group& grid,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,   // fp32 for precision
    const float* __restrict__ v_cache,   // fp32 for precision
    float* __restrict__ attn_out,
    int cache_len,
    int max_seq_len,
    float attn_scale
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    __shared__ float s_max_score[CP_NUM_WARPS_KERNEL];
    __shared__ float s_sum_exp[CP_NUM_WARPS_KERNEL];
    __shared__ float s_out_acc[CP_NUM_WARPS_KERNEL][CP_HEAD_DIM];

    int heads_per_block = (CP_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int head_start = block_id * heads_per_block;
    int head_end = min(head_start + heads_per_block, CP_NUM_Q_HEADS);

    for (int qh = head_start; qh < head_end; qh++) {
        int kv_head = qh / (CP_NUM_Q_HEADS / CP_NUM_KV_HEADS);
        const float* q_head = q + qh * CP_HEAD_DIM;
        float* out_head = attn_out + qh * CP_HEAD_DIM;

        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int pos = warp_id; pos < cache_len; pos += CP_NUM_WARPS_KERNEL) {
            const float* k_pos = k_cache + kv_head * max_seq_len * CP_HEAD_DIM + pos * CP_HEAD_DIM;
            const float* v_pos = v_cache + kv_head * max_seq_len * CP_HEAD_DIM + pos * CP_HEAD_DIM;

            float score = 0.0f;
            for (int d = lane_id; d < CP_HEAD_DIM; d += WARP_SIZE) {
                score += q_head[d] * __ldg(k_pos + d);  // fp32 cache read
            }
            score = cp_warp_reduce_sum_kernel(score) * attn_scale;
            score = __shfl_sync(0xffffffff, score, 0);

            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_diff = expf(old_max - max_score);
            sum_exp = sum_exp * exp_diff + expf(score - max_score);

            float weight = expf(score - max_score);
            #pragma unroll
            for (int d = lane_id, j = 0; d < CP_HEAD_DIM; d += WARP_SIZE, j++) {
                out_acc[j] = out_acc[j] * exp_diff + weight * __ldg(v_pos + d);  // fp32 cache read
            }
        }

        if (lane_id == 0) {
            s_max_score[warp_id] = max_score;
            s_sum_exp[warp_id] = sum_exp;
        }
        #pragma unroll
        for (int d = lane_id, j = 0; d < CP_HEAD_DIM; d += WARP_SIZE, j++) {
            s_out_acc[warp_id][d] = out_acc[j];
        }
        __syncthreads();

        if (warp_id == 0) {
            float global_max = s_max_score[0];
            for (int w = 1; w < CP_NUM_WARPS_KERNEL; w++) {
                if (s_max_score[w] > -INFINITY) {
                    global_max = fmaxf(global_max, s_max_score[w]);
                }
            }

            float total_sum_exp = 0.0f;
            float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int w = 0; w < CP_NUM_WARPS_KERNEL; w++) {
                if (s_max_score[w] > -INFINITY) {
                    float scale_factor = expf(s_max_score[w] - global_max);
                    total_sum_exp += s_sum_exp[w] * scale_factor;

                    #pragma unroll
                    for (int d = lane_id, j = 0; d < CP_HEAD_DIM; d += WARP_SIZE, j++) {
                        final_out[j] += s_out_acc[w][d] * scale_factor;
                    }
                }
            }

            #pragma unroll
            for (int d = lane_id, j = 0; d < CP_HEAD_DIM; d += WARP_SIZE, j++) {
                out_head[d] = final_out[j] / total_sum_exp;
            }
        }
        __syncthreads();
    }

    grid.sync();
}

// =============================================================================
// O Projection + Residual + PostNorm + MLP
// =============================================================================

__device__ void cp_o_proj_postnorm_mlp(
    cg::grid_group& grid,
    const __nv_bfloat16* __restrict__ o_weight,
    const __nv_bfloat16* __restrict__ post_norm_weight,
    const __nv_bfloat16* __restrict__ gate_weight,
    const __nv_bfloat16* __restrict__ up_weight,
    const __nv_bfloat16* __restrict__ down_weight,
    const float* __restrict__ attn_out,
    float* __restrict__ g_residual,
    float* __restrict__ g_activations,
    float* __restrict__ g_mlp_intermediate,
    float* __restrict__ hidden_out  // fp32 output
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // O Projection + Residual
    int hid_per_block = (CP_HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hid_start = block_id * hid_per_block;
    int hid_end = min(hid_start + hid_per_block, CP_HIDDEN_SIZE);

    for (int m_base = hid_start; m_base < hid_end; m_base += CP_NUM_WARPS_KERNEL) {
        int m = m_base + warp_id;

        if (m < hid_end) {
            const __nv_bfloat16* o_row = o_weight + m * CP_Q_SIZE;

            float sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < CP_Q_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(o_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum += __bfloat162float(w_ptr[0]) * attn_out[k] +
                       __bfloat162float(w_ptr[1]) * attn_out[k+1] +
                       __bfloat162float(w_ptr[2]) * attn_out[k+2] +
                       __bfloat162float(w_ptr[3]) * attn_out[k+3];
            }

            sum = cp_warp_reduce_sum_kernel(sum);
            if (lane_id == 0) {
                g_activations[m] = sum + g_residual[m];
            }
        }
    }

    grid.sync();

    // Post-attention RMSNorm
    if (block_id == 0) {
        __shared__ float smem_reduce[CP_NUM_WARPS_KERNEL];

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < CP_HIDDEN_SIZE; i += CP_BLOCK_SIZE) {
            float v = g_activations[i];
            g_residual[i] = v;
            local_sum_sq += v * v;
        }

        local_sum_sq = cp_warp_reduce_sum_kernel(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < CP_NUM_WARPS_KERNEL) ? smem_reduce[lane_id] : 0.0f;
            sum = cp_warp_reduce_sum_kernel(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(CP_HIDDEN_SIZE) + CP_RMS_NORM_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x; i < CP_HIDDEN_SIZE; i += CP_BLOCK_SIZE) {
            float w = __bfloat162float(__ldg(post_norm_weight + i));
            g_activations[i] = g_residual[i] * rstd * w;
        }
    }

    grid.sync();

    // Gate + Up + SiLU
    int int_per_block = (CP_INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
    int int_start = block_id * int_per_block;
    int int_end = min(int_start + int_per_block, CP_INTERMEDIATE_SIZE);

    for (int m_base = int_start; m_base < int_end; m_base += CP_NUM_WARPS_KERNEL) {
        int m = m_base + warp_id;

        if (m < int_end) {
            const __nv_bfloat16* gate_row = gate_weight + m * CP_HIDDEN_SIZE;
            const __nv_bfloat16* up_row = up_weight + m * CP_HIDDEN_SIZE;

            float gate_sum = 0.0f, up_sum = 0.0f;

            #pragma unroll 8
            for (int k = lane_id * 4; k < CP_HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 g_u2 = __ldg(reinterpret_cast<const uint2*>(gate_row + k));
                uint2 u_u2 = __ldg(reinterpret_cast<const uint2*>(up_row + k));
                __nv_bfloat16* g_ptr = reinterpret_cast<__nv_bfloat16*>(&g_u2);
                __nv_bfloat16* u_ptr = reinterpret_cast<__nv_bfloat16*>(&u_u2);

                gate_sum += __bfloat162float(g_ptr[0]) * g_activations[k] +
                            __bfloat162float(g_ptr[1]) * g_activations[k+1] +
                            __bfloat162float(g_ptr[2]) * g_activations[k+2] +
                            __bfloat162float(g_ptr[3]) * g_activations[k+3];

                up_sum += __bfloat162float(u_ptr[0]) * g_activations[k] +
                          __bfloat162float(u_ptr[1]) * g_activations[k+1] +
                          __bfloat162float(u_ptr[2]) * g_activations[k+2] +
                          __bfloat162float(u_ptr[3]) * g_activations[k+3];
            }

            gate_sum = cp_warp_reduce_sum_kernel(gate_sum);
            up_sum = cp_warp_reduce_sum_kernel(up_sum);

            if (lane_id == 0) {
                g_mlp_intermediate[m] = cp_silu(gate_sum) * up_sum;
            }
        }
    }

    grid.sync();

    // Down projection + residual
    for (int m_base = hid_start; m_base < hid_end; m_base += CP_NUM_WARPS_KERNEL) {
        int m = m_base + warp_id;

        if (m < hid_end) {
            const __nv_bfloat16* down_row = down_weight + m * CP_INTERMEDIATE_SIZE;

            float sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < CP_INTERMEDIATE_SIZE; k += WARP_SIZE * 4) {
                uint2 d_u2 = __ldg(reinterpret_cast<const uint2*>(down_row + k));
                __nv_bfloat16* d_ptr = reinterpret_cast<__nv_bfloat16*>(&d_u2);

                sum += __bfloat162float(d_ptr[0]) * g_mlp_intermediate[k] +
                       __bfloat162float(d_ptr[1]) * g_mlp_intermediate[k+1] +
                       __bfloat162float(d_ptr[2]) * g_mlp_intermediate[k+2] +
                       __bfloat162float(d_ptr[3]) * g_mlp_intermediate[k+3];
            }

            sum = cp_warp_reduce_sum_kernel(sum);
            if (lane_id == 0) {
                hidden_out[m] = sum + g_residual[m];  // fp32 output
            }
        }
    }

    grid.sync();
}

// =============================================================================
// Main Code Predictor Decode Kernel
// =============================================================================

__global__ void __launch_bounds__(CP_BLOCK_SIZE, 1)
cp_decode_kernel_with_embedding(
    const __nv_bfloat16* __restrict__ input_embedding,
    const CPLayerWeights* __restrict__ layer_weights,
    const __nv_bfloat16* __restrict__ final_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    float* __restrict__ k_cache,        // fp32 for precision
    float* __restrict__ v_cache,        // fp32 for precision
    float* __restrict__ hidden_buffer,  // fp32 between layers
    float* __restrict__ g_activations,
    float* __restrict__ g_residual,
    float* __restrict__ g_q,
    float* __restrict__ g_k,
    float* __restrict__ g_v,
    float* __restrict__ g_attn_out,
    float* __restrict__ g_mlp_intermediate,
    float* __restrict__ g_normalized,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale
) {
    cg::grid_group grid = cg::this_grid();
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;

    // Copy input embedding (bf16 → fp32)
    for (int i = block_id * CP_BLOCK_SIZE + threadIdx.x; i < CP_HIDDEN_SIZE; i += num_blocks * CP_BLOCK_SIZE) {
        hidden_buffer[i] = __bfloat162float(__ldg(input_embedding + i));
    }
    grid.sync();

    int kv_cache_layer_stride = CP_NUM_KV_HEADS * max_seq_len * CP_HEAD_DIM;

    for (int layer = 0; layer < num_layers; layer++) {
        const CPLayerWeights& w = layer_weights[layer];
        float* layer_k_cache = k_cache + layer * kv_cache_layer_stride;  // fp32
        float* layer_v_cache = v_cache + layer * kv_cache_layer_stride;  // fp32

        cp_matvec_qkv(
            grid, hidden_buffer, w.input_layernorm_weight,
            w.q_proj_weight, w.k_proj_weight, w.v_proj_weight,
            g_activations, g_residual, g_q, g_k, g_v
        );

        cp_qk_norm_rope_cache(
            grid, g_q, g_k, g_v,
            w.q_norm_weight, w.k_norm_weight,
            cos_table, sin_table,
            layer_k_cache, layer_v_cache,
            position, max_seq_len
        );

        cp_attention(
            grid, g_q, layer_k_cache, layer_v_cache, g_attn_out,
            cache_len, max_seq_len, attn_scale
        );

        cp_o_proj_postnorm_mlp(
            grid, w.o_proj_weight, w.post_attn_layernorm_weight,
            w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
            g_attn_out, g_residual, g_activations, g_mlp_intermediate,
            hidden_buffer
        );
    }

    // Final RMSNorm
    if (block_id == 0) {
        __shared__ float smem_reduce[CP_NUM_WARPS_KERNEL];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < CP_HIDDEN_SIZE; i += CP_BLOCK_SIZE) {
            float v = hidden_buffer[i];  // fp32 read
            g_activations[i] = v;
            local_sum_sq += v * v;
        }

        local_sum_sq = cp_warp_reduce_sum_kernel(local_sum_sq);
        if (lane_id == 0) {
            smem_reduce[warp_id] = local_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            float sum = (lane_id < CP_NUM_WARPS_KERNEL) ? smem_reduce[lane_id] : 0.0f;
            sum = cp_warp_reduce_sum_kernel(sum);
            if (lane_id == 0) {
                smem_reduce[0] = rsqrtf(sum / float(CP_HIDDEN_SIZE) + CP_RMS_NORM_EPS);
            }
        }
        __syncthreads();

        float rstd = smem_reduce[0];

        for (int i = threadIdx.x; i < CP_HIDDEN_SIZE; i += CP_BLOCK_SIZE) {
            float wt = __bfloat162float(__ldg(final_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * wt;
        }
    }
}

// =============================================================================
// Launch function
// =============================================================================

extern "C" void launch_cp_decode_with_embedding(
    const void* input_embedding,
    const CPLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream
) {
    void* kernel_args[] = {
        (void*)&input_embedding,
        (void*)&layer_weights,
        (void*)&final_norm_weight,
        (void*)&cos_table,
        (void*)&sin_table,
        (void*)&k_cache,
        (void*)&v_cache,
        (void*)&hidden_buffer,
        (void*)&g_activations,
        (void*)&g_residual,
        (void*)&g_q,
        (void*)&g_k,
        (void*)&g_v,
        (void*)&g_attn_out,
        (void*)&g_mlp_intermediate,
        (void*)&g_normalized,
        (void*)&num_layers,
        (void*)&position,
        (void*)&cache_len,
        (void*)&max_seq_len,
        (void*)&attn_scale
    };

    cudaLaunchCooperativeKernel(
        (void*)cp_decode_kernel_with_embedding,
        dim3(CP_NUM_BLOCKS),
        dim3(CP_BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );
}
