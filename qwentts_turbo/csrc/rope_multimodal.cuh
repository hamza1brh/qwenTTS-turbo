#pragma once

#include "config.cuh"

// =============================================================================
// Multimodal RoPE (Rotary Position Embeddings) for Qwen3-TTS
// =============================================================================
//
// Qwen3-TTS uses multimodal 3D RoPE with interleaved format for the talker backbone.
// This differs from standard 1D RoPE used in regular LLMs.
//
// mrope_section = [24, 20, 20] for temporal/height/width modalities
// Total = 64 = half_dim (since head_dim = 128)
//
// In interleaved mode, the cos/sin values are scattered across the head dimension:
//   cos[i] corresponds to modality (i % 3) at position (i / 3) within that modality
//   Example: [t0, h0, w0, t1, h1, w1, t2, h2, w2, ...]
//
// The rotation formula (same as standard RoPE):
//   out[..., :half] = x[..., :half] * cos - x[..., half:] * sin
//   out[..., half:] = x[..., half:] * cos + x[..., :half] * sin
//
// =============================================================================

// Multimodal RoPE configuration for Qwen3-TTS
constexpr int MROPE_TEMPORAL = 24;
constexpr int MROPE_HEIGHT = 20;
constexpr int MROPE_WIDTH = 20;
constexpr int MROPE_MODALITIES = 3;
constexpr int MROPE_TOTAL = MROPE_TEMPORAL + MROPE_HEIGHT + MROPE_WIDTH;  // 64

// -----------------------------------------------------------------------------
// Precompute interleaved cos/sin tables for multimodal RoPE
// Input: cos_raw, sin_raw of shape [3, max_seq_len, section_dim] for each modality
// Output: cos_interleaved, sin_interleaved of shape [max_seq_len, head_dim]
//
// This function should be called on CPU/Python side to prepare the tables.
// The CUDA kernel expects pre-interleaved tables.
// -----------------------------------------------------------------------------

// Note: The interleaving logic (from PyTorch):
// For interleaved mode with mrope_section = [24, 20, 20]:
//   cos_interleaved[i] = cos_raw[i % 3, pos, i / 3]  (for i < half_dim)
//   Then duplicated for second half
//
// The apply_interleaved_rope function in Python does:
//   x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
//
// This means for half_dim=64 with 3 modalities:
//   indices 0,3,6,...,63 come from temporal (modality 0) - 22 values
//   indices 1,4,7,...,64 come from height (modality 1) - 21 values
//   indices 2,5,8,...,62 come from width (modality 2) - 21 values
//
// But mrope_section=[24,20,20] further restricts which positions are used per modality.

// -----------------------------------------------------------------------------
// Apply multimodal interleaved RoPE to Q and K heads
// Assumes cos/sin are already interleaved to shape [max_seq_len, head_dim]
// -----------------------------------------------------------------------------
__device__ __forceinline__ void rope_multimodal_qk_at_position(
    float* __restrict__ smem_q,              // [num_q_heads * head_dim]
    float* __restrict__ smem_k,              // [num_kv_heads * head_dim]
    const __nv_bfloat16* __restrict__ cos_table,  // [max_seq_len, head_dim] interleaved
    const __nv_bfloat16* __restrict__ sin_table,  // [max_seq_len, head_dim] interleaved
    int position,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int half_dim = head_dim / 2;  // 64 when head_dim=128
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Get cos/sin for this position
    const __nv_bfloat16* cos_pos = cos_table + position * head_dim;
    const __nv_bfloat16* sin_pos = sin_table + position * head_dim;

    // Load cos/sin into registers (head_dim=128, half=64)
    // Each thread in warp loads 2 values (64/32=2)
    float cos_reg[2], sin_reg[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int idx = lane_id * 2 + i;
        if (idx < half_dim) {
            cos_reg[i] = __bfloat162float(__ldg(cos_pos + idx));
            sin_reg[i] = __bfloat162float(__ldg(sin_pos + idx));
        }
    }

    // Process Q heads
    int total_q_heads = num_q_heads;
    int q_heads_per_warp = (total_q_heads + NUM_WARPS - 1) / NUM_WARPS;
    int q_head_start = warp_id * q_heads_per_warp;
    int q_head_end = min(q_head_start + q_heads_per_warp, total_q_heads);

    for (int h = q_head_start; h < q_head_end; h++) {
        float* head_vec = smem_q + h * head_dim;

        // Load the pair values we need for rotation
        // rotate_half: [x0..x63] -> [-x64..-x127, x0..x63]
        // So for output[i] where i < half_dim: out = x[i] * cos - x[i+half] * sin
        // And for output[i] where i >= half_dim: out = x[i] * cos + x[i-half] * sin

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = lane_id * 2 + i;
            if (idx < half_dim) {
                float x0 = head_vec[idx];
                float x1 = head_vec[idx + half_dim];

                // Apply RoPE rotation
                // out[idx] = x0 * cos - x1 * sin
                // out[idx + half_dim] = x1 * cos + x0 * sin
                head_vec[idx] = x0 * cos_reg[i] - x1 * sin_reg[i];
                head_vec[idx + half_dim] = x1 * cos_reg[i] + x0 * sin_reg[i];
            }
        }
    }

    __syncthreads();

    // Process K heads
    int total_kv_heads = num_kv_heads;
    int kv_heads_per_warp = (total_kv_heads + NUM_WARPS - 1) / NUM_WARPS;
    int kv_head_start = warp_id * kv_heads_per_warp;
    int kv_head_end = min(kv_head_start + kv_heads_per_warp, total_kv_heads);

    for (int h = kv_head_start; h < kv_head_end; h++) {
        float* head_vec = smem_k + h * head_dim;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = lane_id * 2 + i;
            if (idx < half_dim) {
                float x0 = head_vec[idx];
                float x1 = head_vec[idx + half_dim];

                head_vec[idx] = x0 * cos_reg[i] - x1 * sin_reg[i];
                head_vec[idx + half_dim] = x1 * cos_reg[i] + x0 * sin_reg[i];
            }
        }
    }

    __syncthreads();
}

// -----------------------------------------------------------------------------
// Multimodal RoPE with QK Norm integration (matches ldg_qk_norm_rope_cache)
// Performs: RMSNorm on Q/K heads -> RoPE -> KV cache update
// -----------------------------------------------------------------------------
__device__ void multimodal_qk_norm_rope_cache(
    cg::grid_group& grid,
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,  // [max_seq_len, head_dim] interleaved
    const __nv_bfloat16* __restrict__ sin_table,  // [max_seq_len, head_dim] interleaved
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int position,
    int max_seq_len
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const __nv_bfloat16* cos_pos = cos_table + position * HEAD_DIM;
    const __nv_bfloat16* sin_pos = sin_table + position * HEAD_DIM;
    int half_dim = HEAD_DIM / 2;

    // Process Q heads
    int q_heads_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int q_head_start = block_id * q_heads_per_block;
    int q_head_end = min(q_head_start + q_heads_per_block, NUM_Q_HEADS);

    for (int h = q_head_start + warp_id; h < q_head_end; h += (blockDim.x / WARP_SIZE)) {
        float* q_head = q + h * HEAD_DIM;

        // RMSNorm on Q head
        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            sum_sq += q_head[i] * q_head[i];
        }

        // Warp reduce
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + RMS_NORM_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        // Apply norm + RoPE in one pass
        // Store normalized values, then apply rotation
        float q_local[HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            q_local[j] = q_head[i] * scale * __bfloat162float(__ldg(q_norm_weight + i));
        }

        // Apply RoPE rotation
        #pragma unroll
        for (int i = lane_id, j = 0; i < half_dim; i += WARP_SIZE, j++) {
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));

            // Get the pair element via shuffle
            int pair_idx = i + half_dim;
            int pair_j = pair_idx / WARP_SIZE;
            float x0 = q_local[j];
            float x1 = __shfl_sync(0xffffffff, q_local[pair_j], pair_idx % WARP_SIZE);

            q_head[i] = x0 * cos_v - x1 * sin_v;
            q_head[i + half_dim] = x1 * cos_v + x0 * sin_v;
        }
    }

    // Process K heads + cache
    int k_heads_per_block = (NUM_KV_HEADS + num_blocks - 1) / num_blocks;
    int k_head_start = block_id * k_heads_per_block;
    int k_head_end = min(k_head_start + k_heads_per_block, NUM_KV_HEADS);

    for (int h = k_head_start + warp_id; h < k_head_end; h += (blockDim.x / WARP_SIZE)) {
        float* k_head = k + h * HEAD_DIM;
        const float* v_head = v + h * HEAD_DIM;
        __nv_bfloat16* k_cache_head = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
        __nv_bfloat16* v_cache_head = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

        // RMSNorm on K head
        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            sum_sq += k_head[i] * k_head[i];
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + RMS_NORM_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float k_local[HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            k_local[j] = k_head[i] * scale * __bfloat162float(__ldg(k_norm_weight + i));
        }

        // Apply RoPE rotation + write to cache
        #pragma unroll
        for (int i = lane_id, j = 0; i < half_dim; i += WARP_SIZE, j++) {
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));

            int pair_idx = i + half_dim;
            int pair_j = pair_idx / WARP_SIZE;
            float x0 = k_local[j];
            float x1 = __shfl_sync(0xffffffff, k_local[pair_j], pair_idx % WARP_SIZE);

            float k_rot0 = x0 * cos_v - x1 * sin_v;
            float k_rot1 = x1 * cos_v + x0 * sin_v;

            k_head[i] = k_rot0;
            k_head[i + half_dim] = k_rot1;
            k_cache_head[i] = __float2bfloat16(k_rot0);
            k_cache_head[i + half_dim] = __float2bfloat16(k_rot1);
        }

        // Write V to cache (no rotation needed)
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            v_cache_head[i] = __float2bfloat16(v_head[i]);
        }
    }

    grid.sync();
}
