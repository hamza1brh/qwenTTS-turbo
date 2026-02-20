# TTS Megakernel Development Log

The journey of adapting MegaQwen's CUDA megakernel from text-only Qwen3-0.6B to **dual-model TTS inference** for Qwen3-TTS-0.6B.

**Final result**: 3.49x speedup over HuggingFace, RTF 0.14 (7x real-time) on RTX 5070 Ti.

---

## The Challenge

Qwen3-TTS is not a single model — it's a **dual-model pipeline**:

```
Text → [Talker Backbone] → group-0 codec token
                         → [Code Predictor] → groups 1-15 codec tokens
                         → combine 16 embeddings → next backbone step
```

The **talker backbone** (28 layers, 1024 hidden, multimodal 3D RoPE) generates one codec token per step. Then a **code predictor** (5 layers, same dims, standard 1D RoPE) generates 15 more tokens autoregressively. Both use GQA (16 Q heads, 8 KV heads, 128 head_dim).

This means each "step" involves 28 backbone layers + 5×15 = 75 code predictor layers = **103 transformer layers per codec frame**.

## Phase 1-3: Foundation (Embedding, Sampling, MatVec)

All three used standard PyTorch operations — no custom CUDA needed:
- **Embedding**: Direct indexing into `codec_embedding.weight`
- **Sampling**: `torch.multinomial` with temperature/top-k/top-p
- **MatVec**: `torch.mv` for codec head projection (exact match)

## Phase 4: Backbone Megakernel

Adapted the existing Qwen3-0.6B megakernel (`fused_decode_ldg.cu`) for the TTS backbone. Key differences from the text LLM:

- **Multimodal 3D RoPE** with `mrope_section = [24, 20, 20]` (temporal/height/width) instead of standard 1D
- During decode, all 3 modalities share the same position, so it reduces to standard RoPE with interleaved frequency ordering
- Created `fused_decode_tts.cu` with `rope_multimodal.cuh`
- Added `decode_step_with_embedding()` — accepts pre-computed combined embedding instead of a token ID

Result: 9/10 token match in first 10 steps. Hidden state diffs < 2.0.

## Phase 5: Code Predictor — PyTorch, Not Megakernel

The code predictor's **short context** (2-16 tokens) made megakernel bf16 precision catastrophic. Each KV entry is 1/3 to 1/16 of total context — any numerical error dominates.

Decision: Use PyTorch CP backbone. It gives 100% match with HF and the compute is small (~7ms per backbone step).

## Phase 6: Full Native Generation Loop

Wired everything together into `NativeTTSGenerator`:
- PyTorch prefill (parallel, fast) → copy KV cache → megakernel decode loop
- 10/10 group-0 token match in first 10 steps with greedy decoding

## Phase 7: Audio Generation + Voice Cloning

Key discoveries:
- **12Hz tokenizer requires `ref_code` prepended** for speaker conditioning — without it, audio is near-silent
- **Greedy decoding produces silence** — model needs `do_sample=True`
- Base model has NO predefined speakers — voice clone is required
- The sampling divergence between megakernel and PyTorch backbone was acceptable: identical prefill, early tokens match, then natural divergence from stochastic sampling

## Phase 8: Profiling — Code Predictor is 73% of Time

Wall-clock breakdown of a single generation step:
```
Backbone decode:    6.4ms  (18%)
Code predictor:    25.5ms  (73%) ← 15 sequential PyTorch forward passes
Embedding combine:  3.1ms  (9%)
```

The code predictor was the clear bottleneck.

## Phase 9: Code Predictor Megakernel — RoPE Bugs

Created `fused_decode_code_predictor.cu` for the CP. Initial results: garbage outputs.

**Root cause**: Two RoPE bugs.

1. **OOB read**: cos/sin tables had 64 entries but were indexed with `head_dim=128`. Reading garbage from beyond the allocation.

2. **Wrong rotation formula**: The kernel used a split-half formula that didn't match PyTorch's `rotate_half`:
   ```
   # PyTorch: x * cos + rotate_half(x) * sin
   # where rotate_half(x) = [-x[64:], x[:64]]
   ```

**Fix**:
- cos/sin tables: `torch.cat((freqs, freqs), dim=-1)` for full 128 dims
- Matched PyTorch's exact rotation formula
- KV cache + hidden buffer: fp32 for precision (short context)

**Result**: 4.26x speedup (5.83ms vs 24.84ms per CP loop), cosine similarity ≥ 0.999.

## Phase 10: Integration — Precision Cascade Failure

Integrated the CP megakernel into the full pipeline. Audio was **silence** (RMS=0.0003) despite 2x wall-clock speedup.

The problem: 0.999 cosine similarity is not enough. The 16-group codec requires **perfectly coherent** token combinations. A 0.1% hidden state error causes different token selection during sampling, which feeds a different embedding back, producing a different group-0 token, which cascades into semantically invalid codec frames.

Evidence: megakernel tokens decoded without ref_code → RMS=0.0004 (silence). PyTorch tokens without ref_code → RMS=0.0907 (audible). The tokens themselves were invalid, not just the decoding.

## The Fix: Repetition Penalty

Instead of chasing fp32 precision throughout the megakernel, we added **repetition penalty** (`penalty=1.2`) to the backbone group-0 sampling. This:
- Prevents the model from getting stuck in degenerate repeated patterns
- Produces diverse, valid codec tokens even with slight numerical differences
- Works with both PyTorch and megakernel CP backends

Combined with the backbone megakernel for decode steps, this gave us audible, high-quality speech.

## Final Architecture

```
PyTorch prefill → copy KV cache → megakernel backbone decode
                                 → megakernel CP decode (per step)
                                 → combine 16 embeddings → next step
```

Both megakernels active. Repetition penalty on group-0 sampling. Voice clone context captured via hooks on `talker.generate` and `model.generate`.

## Results

| Backend | Gen Time | Audio Duration | RTF | Speedup |
|---------|----------|----------------|-----|---------|
| HuggingFace | 2.35s | 4.8s | 0.49 | 1.0x |
| **Dual Megakernel** | **0.67s** | **4.8s** | **0.14** | **3.49x** |

RTF 0.14 = generating audio **7x faster than real-time** on a single RTX 5070 Ti.

GPU stress test (5-minute sustained load):
- VRAM: 3.49 GB model, 3.78 GB peak, 0.28 GB generation overhead
- Throughput: 87.7 codec tokens/sec
- Capacity: 7.18 hours of audio per GPU-hour
- GPU utilization: 86%

## Key Takeaways

1. **Precision requirements differ between text and audio.** Text LLM tolerates bf16 divergence — different tokens still make grammatical sense. Audio codecs require all 16 groups to be coherent; 0.1% error cascades into silence.

2. **The fix isn't always more precision.** The model learned bf16 tie-breaking during training. fp32 projections produced "more correct" tokens that broke the expected trajectory. The right fix was repetition penalty — addressing the symptom (degenerate loops) not the cause (token divergence).

3. **Profile before optimizing.** The backbone megakernel gave 3.2x component speedup but only 1.2x end-to-end because the CP was 73% of time. The CP megakernel was the real win.

4. **Autoregressive error compounding is brutal.** The CP megakernel matched 86.7% in isolated tests but only 6-33% in the actual autoregressive loop. Short-context attention amplifies small precision errors at every step.
