"""
Megakernel TTS decoder for Qwen3-TTS talker backbone.

Provides a Python interface to the fused_decode_tts.cu kernel.
"""

import os
import torch
from torch.utils.cpp_extension import load_inline
import math

_tts_kernel = None

# Model dimensions for Qwen3-TTS-0.6B talker backbone
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072  # From config, though plan shows 2048 for talker
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128  # Explicitly set in model config, NOT hidden_size/num_heads
Q_SIZE = NUM_Q_HEADS * HEAD_DIM   # 2048
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 1024


def _get_cuda_source(filename: str) -> str:
    kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
    with open(os.path.join(kernel_dir, filename)) as f:
        return f.read()


def _compile_tts_kernel():
    global _tts_kernel
    if _tts_kernel is not None:
        return _tts_kernel

    cuda_src = _get_cuda_source("fused_decode_tts.cu")

    cpp_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Must match the struct in fused_decode_tts.cu
struct TTSLayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
};

extern "C" void launch_tts_decode(
    int input_token_id,
    const void* embed_weight,
    const TTSLayerWeights* layer_weights,
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
);

// New: accepts embedding tensor directly (for TTS pipeline integration)
extern "C" void launch_tts_decode_with_embedding(
    const void* input_embedding,
    const TTSLayerWeights* layer_weights,
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
);

class MegakernelTTSDecoder {
public:
    MegakernelTTSDecoder(
        torch::Tensor embed_weight,
        std::vector<torch::Tensor> layer_weights_flat,
        torch::Tensor final_norm_weight,
        torch::Tensor codec_head_weight,
        torch::Tensor cos_table,
        torch::Tensor sin_table,
        int num_layers,
        int max_seq_len
    ) : num_layers_(num_layers), max_seq_len_(max_seq_len) {

        embed_weight_ = embed_weight;
        final_norm_weight_ = final_norm_weight;
        codec_head_weight_ = codec_head_weight;
        cos_table_ = cos_table;
        sin_table_ = sin_table;

        // Store layer weights
        layer_weights_tensors_ = layer_weights_flat;

        // Build layer weights structs
        layer_weights_.resize(num_layers);
        for (int i = 0; i < num_layers; i++) {
            layer_weights_[i].input_layernorm_weight = layer_weights_flat[i * 11 + 0].data_ptr();
            layer_weights_[i].q_proj_weight = layer_weights_flat[i * 11 + 1].data_ptr();
            layer_weights_[i].k_proj_weight = layer_weights_flat[i * 11 + 2].data_ptr();
            layer_weights_[i].v_proj_weight = layer_weights_flat[i * 11 + 3].data_ptr();
            layer_weights_[i].q_norm_weight = layer_weights_flat[i * 11 + 4].data_ptr();
            layer_weights_[i].k_norm_weight = layer_weights_flat[i * 11 + 5].data_ptr();
            layer_weights_[i].o_proj_weight = layer_weights_flat[i * 11 + 6].data_ptr();
            layer_weights_[i].post_attn_layernorm_weight = layer_weights_flat[i * 11 + 7].data_ptr();
            layer_weights_[i].gate_proj_weight = layer_weights_flat[i * 11 + 8].data_ptr();
            layer_weights_[i].up_proj_weight = layer_weights_flat[i * 11 + 9].data_ptr();
            layer_weights_[i].down_proj_weight = layer_weights_flat[i * 11 + 10].data_ptr();
        }

        // Copy layer weights to device
        d_layer_weights_ = torch::empty({num_layers * (int)sizeof(TTSLayerWeights)},
                                         torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_layer_weights_.data_ptr(), layer_weights_.data(),
                   num_layers * sizeof(TTSLayerWeights), cudaMemcpyHostToDevice);

        // Allocate KV cache
        int kv_heads = 8;
        int head_dim = 128;
        k_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kBFloat16).device(torch::kCUDA));

        // Allocate intermediate buffers
        int hidden_size = 1024;
        int q_size = 16 * 128;
        int kv_size = 8 * 128;
        int intermediate_size = 3072;

        hidden_buffer_ = torch::empty({hidden_size}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        g_activations_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_residual_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_q_ = torch::empty({q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_k_ = torch::empty({kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_v_ = torch::empty({kv_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_attn_out_ = torch::empty({q_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_mlp_intermediate_ = torch::empty({intermediate_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        g_normalized_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

        position_ = 0;
        attn_scale_ = 1.0f / sqrtf(128.0f);
    }

    torch::Tensor decode_step(int input_token_id) {
        // Returns the final hidden states (for codec head projection)
        int cache_len = position_ + 1;

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        launch_tts_decode(
            input_token_id,
            embed_weight_.data_ptr(),
            (const TTSLayerWeights*)d_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_buffer_.data_ptr(),
            g_activations_.data_ptr(),
            g_residual_.data_ptr(),
            g_q_.data_ptr(),
            g_k_.data_ptr(),
            g_v_.data_ptr(),
            g_attn_out_.data_ptr(),
            g_mlp_intermediate_.data_ptr(),
            g_normalized_.data_ptr(),
            num_layers_,
            position_,
            cache_len,
            max_seq_len_,
            attn_scale_,
            stream
        );

        position_++;

        // Return hidden states for codec head
        return g_normalized_.clone();
    }

    int decode_step_with_argmax(int input_token_id) {
        // Convenience method: decode + codec head argmax
        torch::Tensor hidden = decode_step(input_token_id);

        // Codec head projection: [hidden_size] @ [vocab_size, hidden_size].T
        // Convert hidden to bfloat16 to match codec_head_weight dtype
        torch::Tensor logits = torch::matmul(hidden.to(torch::kBFloat16), codec_head_weight_.t());
        return logits.argmax().item<int>();
    }

    // New: accepts embedding tensor directly (for TTS pipeline integration)
    torch::Tensor decode_step_with_embedding(torch::Tensor input_embedding) {
        // input_embedding: [hidden_size] or [1, hidden_size] bfloat16
        if (input_embedding.dim() == 2) {
            input_embedding = input_embedding.squeeze(0);
        }
        input_embedding = input_embedding.to(torch::kBFloat16).contiguous();

        int cache_len = position_ + 1;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        launch_tts_decode_with_embedding(
            input_embedding.data_ptr(),
            (const TTSLayerWeights*)d_layer_weights_.data_ptr(),
            final_norm_weight_.data_ptr(),
            cos_table_.data_ptr(),
            sin_table_.data_ptr(),
            k_cache_.data_ptr(),
            v_cache_.data_ptr(),
            hidden_buffer_.data_ptr(),
            g_activations_.data_ptr(),
            g_residual_.data_ptr(),
            g_q_.data_ptr(),
            g_k_.data_ptr(),
            g_v_.data_ptr(),
            g_attn_out_.data_ptr(),
            g_mlp_intermediate_.data_ptr(),
            g_normalized_.data_ptr(),
            num_layers_,
            position_,
            cache_len,
            max_seq_len_,
            attn_scale_,
            stream
        );

        position_++;
        return g_normalized_.clone();
    }

    void reset() {
        position_ = 0;
        k_cache_.zero_();
        v_cache_.zero_();
    }

    int position() const { return position_; }

    void set_position(int pos) {
        position_ = pos;
    }

    torch::Tensor get_hidden_states() const { return g_normalized_.clone(); }
    torch::Tensor get_k_cache() const { return k_cache_; }
    torch::Tensor get_v_cache() const { return v_cache_; }

    void set_kv_cache(torch::Tensor k_cache, torch::Tensor v_cache) {
        // Copy KV cache from external source (e.g., PyTorch DynamicCache)
        // Expected shape: [num_layers, num_kv_heads, seq_len, head_dim]
        k_cache_.index_put_({torch::indexing::Slice(), torch::indexing::Slice(),
                             torch::indexing::Slice(0, k_cache.size(2)), torch::indexing::Slice()},
                            k_cache.to(torch::kBFloat16));
        v_cache_.index_put_({torch::indexing::Slice(), torch::indexing::Slice(),
                             torch::indexing::Slice(0, v_cache.size(2)), torch::indexing::Slice()},
                            v_cache.to(torch::kBFloat16));
    }

private:
    int num_layers_;
    int max_seq_len_;
    int position_;
    float attn_scale_;

    torch::Tensor embed_weight_;
    torch::Tensor final_norm_weight_;
    torch::Tensor codec_head_weight_;
    torch::Tensor cos_table_;
    torch::Tensor sin_table_;
    torch::Tensor d_layer_weights_;

    std::vector<torch::Tensor> layer_weights_tensors_;
    std::vector<TTSLayerWeights> layer_weights_;

    torch::Tensor k_cache_, v_cache_;
    torch::Tensor hidden_buffer_, g_activations_, g_residual_;
    torch::Tensor g_q_, g_k_, g_v_, g_attn_out_;
    torch::Tensor g_mlp_intermediate_, g_normalized_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MegakernelTTSDecoder>(m, "MegakernelTTSDecoder")
        .def(py::init<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor, int, int>())
        .def("decode_step", &MegakernelTTSDecoder::decode_step)
        .def("decode_step_with_argmax", &MegakernelTTSDecoder::decode_step_with_argmax)
        .def("decode_step_with_embedding", &MegakernelTTSDecoder::decode_step_with_embedding)
        .def("reset", &MegakernelTTSDecoder::reset)
        .def("position", &MegakernelTTSDecoder::position)
        .def("set_position", &MegakernelTTSDecoder::set_position)
        .def("get_hidden_states", &MegakernelTTSDecoder::get_hidden_states)
        .def("get_k_cache", &MegakernelTTSDecoder::get_k_cache)
        .def("get_v_cache", &MegakernelTTSDecoder::get_v_cache)
        .def("set_kv_cache", &MegakernelTTSDecoder::set_kv_cache);
}
"""

    kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

    # Use PTX compilation for forward compatibility with newer GPUs
    # sm_89 PTX can JIT compile to sm_120 (RTX 5070 Ti) at runtime
    _tts_kernel = load_inline(
        name="megakernel_tts",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            # Compile to PTX for forward compatibility
            "-gencode", "arch=compute_89,code=compute_89",
            "--expt-relaxed-constexpr",
            "-I" + kernel_dir,
        ],
        verbose=False,
    )

    return _tts_kernel


def _extract_tts_backbone_weights(model, max_seq_len: int = 4096) -> dict:
    """Extract weights from talker backbone for the megakernel."""
    talker = model.talker
    backbone = talker.model
    num_layers = len(backbone.layers)
    rope_theta = backbone.config.rope_theta  # typically 1000000.0

    # Embedding and head weights
    embed_weight = backbone.codec_embedding.weight.contiguous()
    codec_head_weight = talker.codec_head.weight.contiguous()
    final_norm_weight = backbone.norm.weight.contiguous()
    vocab_size = embed_weight.shape[0]

    # Layer weights (11 per layer, matching C++ struct order)
    layer_weights = []
    for layer_idx in range(num_layers):
        layer = backbone.layers[layer_idx]
        layer_weights.extend([
            layer.input_layernorm.weight.contiguous(),
            layer.self_attn.q_proj.weight.contiguous(),
            layer.self_attn.k_proj.weight.contiguous(),
            layer.self_attn.v_proj.weight.contiguous(),
            layer.self_attn.q_norm.weight.contiguous(),
            layer.self_attn.k_norm.weight.contiguous(),
            layer.self_attn.o_proj.weight.contiguous(),
            layer.post_attention_layernorm.weight.contiguous(),
            layer.mlp.gate_proj.weight.contiguous(),
            layer.mlp.up_proj.weight.contiguous(),
            layer.mlp.down_proj.weight.contiguous(),
        ])

    # Build RoPE tables — multimodal 3D RoPE but during decode all modalities
    # use the same position, so it reduces to standard 1D RoPE.
    # cos/sin shape: [max_seq_len, head_dim] with concatenated freqs.
    half_dim = HEAD_DIM // 2  # 64
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [max_seq_len, 64]
    emb = torch.cat((freqs, freqs), dim=-1)   # [max_seq_len, 128]
    cos_table = torch.cos(emb).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(emb).to(torch.bfloat16).cuda().contiguous()

    return {
        "embed_weight": embed_weight,
        "layer_weights": layer_weights,
        "final_norm_weight": final_norm_weight,
        "codec_head_weight": codec_head_weight,
        "cos_table": cos_table,
        "sin_table": sin_table,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
    }


class TTSMegakernelGenerator:
    """High-level TTS generator using the megakernel for backbone inference."""

    def __init__(self, model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                 model=None, max_seq_len: int = 4096):
        if model is None:
            from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
            tts = Qwen3TTSModel.from_pretrained(model_path, dtype="bfloat16", device_map="cuda")
            model = tts.model

        print("Extracting backbone weights...")
        weights = _extract_tts_backbone_weights(model, max_seq_len)

        print("Compiling TTS megakernel...")
        kernel = _compile_tts_kernel()

        self.decoder = kernel.MegakernelTTSDecoder(
            weights["embed_weight"],
            weights["layer_weights"],
            weights["final_norm_weight"],
            weights["codec_head_weight"],
            weights["cos_table"],
            weights["sin_table"],
            weights["num_layers"],
            max_seq_len,
        )

        self.max_seq_len = max_seq_len
        self.vocab_size = weights["vocab_size"]

    def decode_step(self, input_token_id: int) -> torch.Tensor:
        """Run one decode step, returning hidden states."""
        return self.decoder.decode_step(input_token_id)

    def decode_step_with_argmax(self, input_token_id: int) -> int:
        """Run one decode step, returning argmax token."""
        return self.decoder.decode_step_with_argmax(input_token_id)

    def decode_step_with_embedding(self, input_embedding: torch.Tensor) -> torch.Tensor:
        """Run one decode step with pre-computed embedding, returning hidden states.

        Args:
            input_embedding: [hidden_size] or [1, hidden_size] bfloat16 tensor

        Returns:
            hidden_states: [hidden_size] float32 tensor (after final RMSNorm)
        """
        return self.decoder.decode_step_with_embedding(input_embedding)

    def reset(self):
        """Reset KV cache and position."""
        self.decoder.reset()

    def set_position(self, pos: int):
        """Set the current position (for resuming from PyTorch prefill)."""
        self.decoder.set_position(pos)

    def set_kv_cache(self, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """Set KV cache from external source (e.g., PyTorch DynamicCache).

        Args:
            k_cache: [num_layers, num_kv_heads, seq_len, head_dim] tensor
            v_cache: [num_layers, num_kv_heads, seq_len, head_dim] tensor
        """
        self.decoder.set_kv_cache(k_cache, v_cache)

    def get_k_cache(self) -> torch.Tensor:
        """Get reference to K cache tensor."""
        return self.decoder.get_k_cache()

    def get_v_cache(self) -> torch.Tensor:
        """Get reference to V cache tensor."""
        return self.decoder.get_v_cache()

    @property
    def position(self) -> int:
        return self.decoder.position()


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--num-tokens", type=int, default=100)
    args = parser.parse_args()

    print("Initializing TTS Megakernel Generator...")
    gen = TTSMegakernelGenerator(args.model_path)

    if args.benchmark:
        print(f"\nBenchmarking {args.num_tokens} decode steps...")

        # Warmup
        for _ in range(10):
            gen.reset()
            for i in range(10):
                gen.decode_step_with_argmax(i % gen.vocab_size)

        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(5):
            gen.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for i in range(args.num_tokens):
                gen.decode_step_with_argmax(i % gen.vocab_size)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        mean_time = sum(times) / len(times)
        tokens_per_sec = args.num_tokens / mean_time

        print(f"\nResults ({args.num_tokens} tokens):")
        print(f"  Mean time: {mean_time * 1000:.2f} ms")
        print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
        print(f"  Per-token latency: {mean_time * 1000 / args.num_tokens:.3f} ms")

    else:
        print("\nTesting single decode step...")
        gen.reset()
        token = 0
        hidden = gen.decode_step(token)
        print(f"Hidden states shape: {hidden.shape}")
        print(f"Hidden states mean: {hidden.mean().item():.6f}")
        print(f"Position: {gen.position}")
