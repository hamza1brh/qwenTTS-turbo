# qwentts-turbo

CUDA megakernel-accelerated inference for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). **3.75x faster** than HuggingFace `generate()`.

RTF 0.13 = generating audio **7x faster than real-time** on RTX 5070 Ti.

## Performance

| Backend | Gen Time | Audio Duration | RTF | Speedup |
|---------|----------|----------------|-----|---------|
| HuggingFace `generate()` | 2.28s | 4.6s | 0.49 | 1.0x |
| vLLM-Omni | 2.28s | 4.6s | 0.49 | 1.0x |
| **Dual Megakernel** | **0.64s** | **4.8s** | **0.13** | **3.75x** |

*3 English prompts × 5 runs each, median values. RTX 5070 Ti, Qwen3-TTS-0.6B. All 30 megakernel outputs audible (RMS 0.10–0.14).*

Both backbone (28 layers) and code predictor (5 layers) are replaced with fused CUDA megakernels — each fuses an entire transformer pass into a single kernel launch using cooperative groups.

vLLM-Omni wraps the same HuggingFace `generate()` — no speedup for single-request inference.

## Environment Setup

Tested and verified with:

| Package | Version |
|---------|---------|
| Python | 3.12 |
| torch | 2.9.1+cu128 |
| transformers | 4.57.3 |
| qwen-tts | 0.1.1 |
| flash-attn | 2.7.4 |
| numpy | 1.26.4 |
| soundfile | 0.12.1 |
| librosa | 0.10.2 |

**GPU**: NVIDIA RTX 5070 Ti (compute capability 12.0), CUDA 12.8

### With uv (recommended)

[uv](https://docs.astral.sh/uv/) handles everything — including pulling PyTorch from the CUDA index automatically.

```bash
git clone https://github.com/hamza1brh/qwentts-turbo
cd qwentts-turbo
uv sync                    # deterministic install from uv.lock
```

For the Gradio demo:

```bash
uv sync --extra demo
```

### With pip

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# PyTorch must come from the CUDA index (default PyPI is CPU-only)
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/hamza1brh/qwentts-turbo
cd qwentts-turbo
pip install -e .
```

### Step 4: Download model weights

```bash
# Option A: auto-download from HuggingFace (default)
# Nothing to do — load_model() downloads Qwen/Qwen3-TTS-0.6B-Preview automatically

# Option B: local checkpoint
export QWEN_TTS_MODEL=/path/to/Qwen3-TTS-12Hz-0.6B-Base
```

### Step 5: Set reference audio for voice cloning

```bash
export QWEN_TTS_REF_AUDIO=/path/to/reference.mp3
```

Or use the bundled test audio in `tests/sample_ref.mp3`.

CUDA kernels are JIT-compiled on first run (~30s). Subsequent runs use cache.

## Quick Start

```python
from qwentts_turbo import load_model, NativeTTSGenerator
from qwentts_turbo import capture_voice_clone_context, decode_to_audio
import soundfile as sf

# Load model
tts = load_model()  # or load_model("/path/to/local/checkpoint")

# Create dual megakernel generator
gen = NativeTTSGenerator(model=tts.model, backend="megakernel", cp_backend="megakernel")

# Capture voice clone context
ctx = capture_voice_clone_context(
    tts, text="Hello world", language="English",
    ref_audio_path="reference.mp3", ref_text="Transcript of the reference audio",
)

# Generate codec tokens
codes = gen.generate(
    ctx["inputs_embeds"], ctx["attention_mask"],
    ctx["trailing_text_hidden"], ctx["tts_pad_embed"],
    max_tokens=300, do_sample=True, temperature=0.9, top_k=50,
    min_new_tokens=20, repetition_penalty=1.2,
)

# Decode to audio
audio, sr = decode_to_audio(
    tts.model.speech_tokenizer, codes,
    ctx.get("ref_code", [None])[0],
)
sf.write("output.wav", audio, sr)
```

## Gradio Demo

```bash
export QWEN_TTS_MODEL=/path/to/Qwen3-TTS-12Hz-0.6B-Base
export QWEN_TTS_REF_AUDIO=/path/to/reference.mp3
python demo.py
```

Compare PyTorch vs megakernel side-by-side with timing stats.

## Running Tests

```bash
# Audio quality gate test (all 4 gates must pass)
python tests/test_audio.py

# Speed benchmark (3 prompts × 5 runs)
python tests/bench.py

# Benchmark vs vLLM-Omni
python tests/bench_vllm.py
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `QWEN_TTS_MODEL` | `Qwen/Qwen3-TTS-0.6B-Preview` | Model path or HuggingFace hub ID |
| `QWEN_TTS_REF_AUDIO` | *(bundled test audio)* | Reference audio for voice cloning |

## How It Works

Qwen3-TTS has a **dual-model architecture**: a 28-layer backbone produces one codec token per frame, then a 5-layer code predictor generates 15 more autoregressively. That's ~9,000 transformer layers for 7 seconds of audio.

qwentts-turbo replaces **both** models with fused CUDA megakernels — each fuses an entire transformer pass (RMSNorm → QKV → RoPE → Attention → MLP, all layers) into a single kernel launch using cooperative groups for grid-wide sync.

```
PyTorch prefill (17ms, once)
         |
         v
    [Decode Loop]
    |
    |-- Backbone Megakernel (2.9ms/step)
    |   28 layers, multimodal 3D RoPE
    |   -> group 0 token
    |
    |-- CP Megakernel (x15 steps, 5.8ms total)
    |   5 layers, standard 1D RoPE, fp32 KV cache
    |   -> groups 1-15
    |
    |-- Combine 16 embeddings -> next input
    |
    v
DAC decode -> audio
```

See [DEVLOG.md](DEVLOG.md) for the full development story — RoPE bugs, precision cascades, and how cooperative groups enable grid-wide synchronization.

## Project Structure

```
qwentts-turbo/
├── qwentts_turbo/
│   ├── generator.py              # NativeTTSGenerator - main generation loop
│   ├── megakernel_backbone.py    # 28-layer backbone megakernel wrapper
│   ├── megakernel_code_predictor.py  # 5-layer CP megakernel wrapper
│   ├── utils.py                  # Model loading, voice clone capture, audio decode
│   └── csrc/                     # CUDA kernels (JIT compiled)
│       ├── fused_decode_tts.cu
│       ├── fused_decode_code_predictor.cu
│       ├── config.cuh
│       └── rope_multimodal.cuh
├── demo.py                       # Gradio web UI
├── tests/
│   ├── test_audio.py             # Audio quality gate test
│   ├── bench.py                  # Speed benchmark
│   └── bench_vllm.py             # HF vs megakernel vs vLLM-Omni comparison
├── DEVLOG.md                     # Development log
└── pyproject.toml
```

## Credits

- CUDA megakernel architecture from [Infatoshi/MegaQwen](https://github.com/Infatoshi/MegaQwen)
- TTS dual megakernel by [hamza1brh](https://github.com/hamza1brh)
- Model: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Cloud

## License

MIT
