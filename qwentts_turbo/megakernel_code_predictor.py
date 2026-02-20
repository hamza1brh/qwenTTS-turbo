"""
Megakernel Code Predictor decoder for Qwen3-TTS.

Key differences from backbone megakernel:
- 5 layers instead of 28
- Standard 1D RoPE (not multimodal interleaved)
- Same dimensions otherwise (1024 hidden, 16 Q heads, 8 KV heads, 128 head_dim)
"""

import os
import torch
import math
from torch.utils.cpp_extension import load_inline
from typing import Optional

_cp_kernel = None

# Model dimensions for Qwen3-TTS code predictor
CP_HIDDEN_SIZE = 1024
CP_INTERMEDIATE_SIZE = 3072
CP_NUM_Q_HEADS = 16
CP_NUM_KV_HEADS = 8
CP_HEAD_DIM = 128
CP_Q_SIZE = CP_NUM_Q_HEADS * CP_HEAD_DIM   # 2048
CP_KV_SIZE = CP_NUM_KV_HEADS * CP_HEAD_DIM  # 1024
CP_NUM_LAYERS = 5


def _get_cuda_source(filename: str) -> str:
    kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
    with open(os.path.join(kernel_dir, filename)) as f:
        return f.read()


def _compile_cp_kernel():
    global _cp_kernel
    if _cp_kernel is not None:
        return _cp_kernel

    cuda_src = _get_cuda_source("fused_decode_code_predictor.cu")

    cpp_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

struct CPLayerWeights {
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
);

class MegakernelCodePredictorDecoder {
public:
    MegakernelCodePredictorDecoder(
        std::vector<torch::Tensor> layer_weights_flat,
        torch::Tensor final_norm_weight,
        torch::Tensor cos_table,
        torch::Tensor sin_table,
        int num_layers,
        int max_seq_len
    ) : num_layers_(num_layers), max_seq_len_(max_seq_len) {

        final_norm_weight_ = final_norm_weight;
        cos_table_ = cos_table;
        sin_table_ = sin_table;

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
        d_layer_weights_ = torch::empty({num_layers * (int)sizeof(CPLayerWeights)},
                                         torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cudaMemcpy(d_layer_weights_.data_ptr(), layer_weights_.data(),
                   num_layers * sizeof(CPLayerWeights), cudaMemcpyHostToDevice);

        // Allocate KV cache (5 layers) - fp32 for precision with short context
        int kv_heads = 8;
        int head_dim = 128;
        k_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kFloat32).device(torch::kCUDA));
        v_cache_ = torch::zeros({num_layers, kv_heads, max_seq_len, head_dim},
                                torch::dtype(torch::kFloat32).device(torch::kCUDA));

        // Allocate intermediate buffers
        int hidden_size = 1024;
        int q_size = 16 * 128;
        int kv_size = 8 * 128;
        int intermediate_size = 3072;

        hidden_buffer_ = torch::empty({hidden_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
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

    torch::Tensor decode_step_with_embedding(torch::Tensor input_embedding) {
        if (input_embedding.dim() == 2) {
            input_embedding = input_embedding.squeeze(0);
        }
        input_embedding = input_embedding.to(torch::kBFloat16).contiguous();

        int cache_len = position_ + 1;
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

        launch_cp_decode_with_embedding(
            input_embedding.data_ptr(),
            (const CPLayerWeights*)d_layer_weights_.data_ptr(),
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
    void set_position(int pos) { position_ = pos; }
    torch::Tensor get_hidden_states() const { return g_normalized_.clone(); }

    void set_kv_cache(torch::Tensor k_cache, torch::Tensor v_cache) {
        // Copy KV cache from external source (e.g., PyTorch DynamicCache)
        // Expected shape: [num_layers, num_kv_heads, seq_len, head_dim]
        // Use fp32 for precision with short context
        k_cache_.index_put_({torch::indexing::Slice(), torch::indexing::Slice(),
                             torch::indexing::Slice(0, k_cache.size(2)), torch::indexing::Slice()},
                            k_cache.to(torch::kFloat32));
        v_cache_.index_put_({torch::indexing::Slice(), torch::indexing::Slice(),
                             torch::indexing::Slice(0, v_cache.size(2)), torch::indexing::Slice()},
                            v_cache.to(torch::kFloat32));
    }

private:
    int num_layers_;
    int max_seq_len_;
    int position_;
    float attn_scale_;

    torch::Tensor final_norm_weight_;
    torch::Tensor cos_table_;
    torch::Tensor sin_table_;
    torch::Tensor d_layer_weights_;

    std::vector<torch::Tensor> layer_weights_tensors_;
    std::vector<CPLayerWeights> layer_weights_;

    torch::Tensor k_cache_, v_cache_;
    torch::Tensor hidden_buffer_, g_activations_, g_residual_;
    torch::Tensor g_q_, g_k_, g_v_, g_attn_out_;
    torch::Tensor g_mlp_intermediate_, g_normalized_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MegakernelCodePredictorDecoder>(m, "MegakernelCodePredictorDecoder")
        .def(py::init<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, int, int>())
        .def("decode_step_with_embedding", &MegakernelCodePredictorDecoder::decode_step_with_embedding)
        .def("reset", &MegakernelCodePredictorDecoder::reset)
        .def("position", &MegakernelCodePredictorDecoder::position)
        .def("set_position", &MegakernelCodePredictorDecoder::set_position)
        .def("get_hidden_states", &MegakernelCodePredictorDecoder::get_hidden_states)
        .def("set_kv_cache", &MegakernelCodePredictorDecoder::set_kv_cache);
}
"""

    kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

    _cp_kernel = load_inline(
        name="megakernel_code_predictor",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3",
            # No --use_fast_math: need precise exp() for softmax with short context
            "-std=c++17",
            "-gencode", "arch=compute_89,code=compute_89",
            "--expt-relaxed-constexpr",
            "-I" + kernel_dir,
        ],
        verbose=False,
    )

    return _cp_kernel


def extract_code_predictor_weights(model_path: str) -> dict:
    """Extract weights from code predictor for megakernel."""
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    print(f"Loading Qwen3-TTS model from {model_path}...")
    model = Qwen3TTSModel.from_pretrained(model_path, dtype="bfloat16", device_map="cuda")
    code_predictor = model.model.talker.code_predictor

    # Extract layer weights
    layer_weights = []
    for layer_idx in range(CP_NUM_LAYERS):
        layer = code_predictor.model.layers[layer_idx]

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

    final_norm_weight = code_predictor.model.norm.weight.contiguous()

    # Build standard 1D RoPE tables (matching PyTorch's format)
    # PyTorch concatenates freqs: torch.cat((freqs, freqs), dim=-1)
    # So cos/sin are [max_seq_len, head_dim] where head_dim=128
    max_seq_len = 4096
    rope_theta = 1000000.0
    half_dim = CP_HEAD_DIM // 2  # 64

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [max_seq_len, 64]

    # Concatenate to match PyTorch: [max_seq_len, 128]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_table = torch.cos(emb).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(emb).to(torch.bfloat16).cuda().contiguous()

    return {
        "layer_weights": layer_weights,
        "final_norm_weight": final_norm_weight,
        "cos_table": cos_table,
        "sin_table": sin_table,
        "num_layers": CP_NUM_LAYERS,
    }


class CodePredictorMegakernelGenerator:
    """High-level code predictor generator using megakernel."""

    def __init__(self, model_path: str, max_seq_len: int = 256):
        print(f"Loading code predictor weights from {model_path}...")
        self.weights = extract_code_predictor_weights(model_path)

        print("Compiling code predictor megakernel...")
        kernel = _compile_cp_kernel()

        self.decoder = kernel.MegakernelCodePredictorDecoder(
            self.weights["layer_weights"],
            self.weights["final_norm_weight"],
            self.weights["cos_table"],
            self.weights["sin_table"],
            self.weights["num_layers"],
            max_seq_len,
        )

        self.max_seq_len = max_seq_len

    def decode_step_with_embedding(self, input_embedding: torch.Tensor) -> torch.Tensor:
        """Run one decode step, returning hidden states."""
        return self.decoder.decode_step_with_embedding(input_embedding)

    def reset(self):
        """Reset KV cache and position."""
        self.decoder.reset()

    @property
    def position(self) -> int:
        return self.decoder.position()


if __name__ == "__main__":
    import time

    MODEL_PATH = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")

    print("Initializing Code Predictor Megakernel...")
    gen = CodePredictorMegakernelGenerator(MODEL_PATH)

    # Test with random embedding
    print("\nTesting single decode step...")
    gen.reset()
    test_embed = torch.randn(1, CP_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    hidden = gen.decode_step_with_embedding(test_embed)
    print(f"Hidden states shape: {hidden.shape}")
    print(f"Hidden states mean: {hidden.mean().item():.6f}")
    print(f"Position: {gen.position}")

    # Benchmark
    print(f"\nBenchmarking 15 decode steps (like code predictor autoregressive loop)...")
    times = []
    for _ in range(10):
        gen.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(15):
            hidden = gen.decode_step_with_embedding(test_embed)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    mean_time = sum(times) / len(times)
    print(f"Mean time for 15 steps: {mean_time * 1000:.2f}ms")
    print(f"Per-step latency: {mean_time * 1000 / 15:.3f}ms")
