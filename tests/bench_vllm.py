#!/usr/bin/env python3
"""
Benchmark: qwentts-turbo megakernel vs HuggingFace generate vs vLLM-Omni.

Tests three backends:
  1. HuggingFace generate() — qwen_tts.Qwen3TTSModel (baseline)
  2. qwentts-turbo dual megakernel
  3. vLLM-Omni Qwen3TTSModel (if available)

Run: python tests/bench_vllm.py
"""
import os
import sys
import time
import types
import torch
import numpy as np
import soundfile as sf

# ── Paths ──────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
_BUNDLED_REF = os.path.join(os.path.dirname(__file__), "sample_ref.mp3")
REF_AUDIO_PATH = os.environ.get("QWEN_TTS_REF_AUDIO", "") or _BUNDLED_REF
REF_TEXT = "This is a reference audio sample for voice cloning"
TEXT = "Technology is changing the world in ways we never imagined possible."
LANGUAGE = "English"
MAX_TOKENS = 200
WARMUP = 2
RUNS = 5
SAMPLE_PARAMS = dict(do_sample=True, temperature=0.9, top_k=50, top_p=1.0)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tts_output", "bench_vllm")


def load_ref_audio():
    audio, sr = sf.read(REF_AUDIO_PATH)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))


# ── Backend 1: HuggingFace generate (qwen_tts) ───────────────────────
def bench_hf(n_warmup, n_runs):
    from qwen_tts import Qwen3TTSModel
    print("  Loading qwen_tts.Qwen3TTSModel...")
    tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, dtype="bfloat16", device_map="cuda")

    # Monkey-patch to suppress validation error
    def patched_validate(self, model_kwargs):
        pass
    tts.model.talker._validate_model_kwargs = types.MethodType(patched_validate, tts.model.talker)

    ref_audio, ref_sr = load_ref_audio()
    gen_kwargs = dict(
        text=TEXT, language=LANGUAGE,
        ref_audio=(ref_audio, ref_sr), ref_text=REF_TEXT,
        max_new_tokens=MAX_TOKENS, min_new_tokens=20,
        repetition_penalty=1.05, **SAMPLE_PARAMS,
    )

    for i in range(n_warmup):
        print(f"    warmup {i+1}/{n_warmup}...")
        tts.generate_voice_clone(**gen_kwargs)

    results = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        wavs, sr = tts.generate_voice_clone(**gen_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        audio = wavs[0]
        dur = len(audio) / sr
        r = rms(audio)
        results.append(dict(gen_s=elapsed, dur_s=dur, rtf=elapsed / dur, rms=r))
        print(f"    run {i+1}: {elapsed:.2f}s → {dur:.2f}s audio, RTF {elapsed/dur:.3f}, RMS {r:.4f}")
        sf.write(os.path.join(OUTPUT_DIR, f"hf_{i}.wav"), audio, sr)

    del tts
    torch.cuda.empty_cache()
    return results


# ── Backend 2: qwentts-turbo megakernel ──────────────────────────────
def bench_mega(n_warmup, n_runs):
    from qwentts_turbo.utils import load_model, capture_voice_clone_context, decode_to_audio
    from qwentts_turbo.generator import NativeTTSGenerator

    print("  Loading qwentts-turbo (megakernel)...")
    tts = load_model(MODEL_PATH)
    gen = NativeTTSGenerator(model=tts.model, backend="megakernel", cp_backend="megakernel")
    gen._speech_tokenizer = tts.model.speech_tokenizer

    print("  Capturing voice clone context...")
    captured = capture_voice_clone_context(
        tts, text=TEXT, language=LANGUAGE,
        ref_audio_path=REF_AUDIO_PATH, ref_text=REF_TEXT,
        max_tokens=MAX_TOKENS, **SAMPLE_PARAMS,
    )
    ref_code = captured.get("ref_code", [None])[0]
    ie = captured["inputs_embeds"]
    am = captured["attention_mask"]
    th = captured["trailing_text_hidden"]
    tp = captured["tts_pad_embed"]

    for i in range(n_warmup):
        print(f"    warmup {i+1}/{n_warmup}...")
        with torch.no_grad():
            gen.generate(ie, am, th, tp, MAX_TOKENS,
                         min_new_tokens=20, repetition_penalty=1.2, **SAMPLE_PARAMS)

    results = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            codes = gen.generate(ie, am, th, tp, MAX_TOKENS,
                                 min_new_tokens=20, repetition_penalty=1.2, **SAMPLE_PARAMS)
        audio, sr = decode_to_audio(gen._speech_tokenizer, codes, ref_code)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        dur = len(audio) / sr
        r = rms(audio)
        results.append(dict(gen_s=elapsed, dur_s=dur, rtf=elapsed / dur, rms=r))
        print(f"    run {i+1}: {elapsed:.2f}s → {dur:.2f}s audio, RTF {elapsed/dur:.3f}, RMS {r:.4f}")
        sf.write(os.path.join(OUTPUT_DIR, f"mega_{i}.wav"), audio, sr)

    del tts, gen
    torch.cuda.empty_cache()
    return results


# ── Backend 3: vLLM-Omni ─────────────────────────────────────────────
def bench_vllm(n_warmup, n_runs):
    try:
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts import (
            Qwen3TTSModel as VLLMQwen3TTSModel,
        )
    except ImportError:
        print("  vLLM-Omni not installed, skipping.")
        return None

    print("  Loading vllm_omni.Qwen3TTSModel...")
    tts = VLLMQwen3TTSModel.from_pretrained(MODEL_PATH, dtype="bfloat16", device_map="cuda")

    # Monkey-patch to suppress validation error (same issue as qwen-tts)
    def patched_validate(self, model_kwargs):
        pass
    tts.model.talker._validate_model_kwargs = types.MethodType(patched_validate, tts.model.talker)

    ref_audio, ref_sr = load_ref_audio()
    gen_kwargs = dict(
        text=TEXT, language=LANGUAGE,
        ref_audio=(ref_audio, ref_sr), ref_text=REF_TEXT,
        max_new_tokens=MAX_TOKENS, min_new_tokens=20,
        repetition_penalty=1.05, **SAMPLE_PARAMS,
    )

    for i in range(n_warmup):
        print(f"    warmup {i+1}/{n_warmup}...")
        tts.generate_voice_clone(**gen_kwargs)

    results = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        wavs, sr = tts.generate_voice_clone(**gen_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        audio = wavs[0]
        dur = len(audio) / sr
        r = rms(audio)
        results.append(dict(gen_s=elapsed, dur_s=dur, rtf=elapsed / dur, rms=r))
        print(f"    run {i+1}: {elapsed:.2f}s → {dur:.2f}s audio, RTF {elapsed/dur:.3f}, RMS {r:.4f}")
        sf.write(os.path.join(OUTPUT_DIR, f"vllm_{i}.wav"), audio, sr)

    del tts
    torch.cuda.empty_cache()
    return results


def summarize(name, results):
    if results is None:
        return None
    gen = np.median([r["gen_s"] for r in results])
    dur = np.median([r["dur_s"] for r in results])
    rtf = np.median([r["rtf"] for r in results])
    rms_val = np.median([r["rms"] for r in results])
    return dict(name=name, gen_s=gen, dur_s=dur, rtf=rtf, rms=rms_val)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Model: {MODEL_PATH}")
    print(f"Ref audio: {REF_AUDIO_PATH}")
    print(f"Text: {TEXT[:60]}...")
    print(f"Warmup: {WARMUP}, Timed runs: {RUNS}")
    print()

    # Run benchmarks sequentially (each loads/unloads the model)
    print("[1/3] HuggingFace generate (qwen_tts)")
    hf_results = bench_hf(WARMUP, RUNS)

    print(f"\n[2/3] qwentts-turbo megakernel")
    mega_results = bench_mega(WARMUP, RUNS)

    print(f"\n[3/3] vLLM-Omni")
    vllm_results = bench_vllm(WARMUP, RUNS)

    # Summary table
    summaries = [
        summarize("HuggingFace", hf_results),
        summarize("Megakernel", mega_results),
        summarize("vLLM-Omni", vllm_results),
    ]
    summaries = [s for s in summaries if s is not None]

    hf_gen = summaries[0]["gen_s"]

    print(f"\n{'='*70}")
    print(f"  Results (median of {RUNS} runs, {WARMUP} warmup)")
    print(f"  {'Backend':<14} {'Gen(s)':>7} {'Audio(s)':>9} {'RTF':>6} {'RMS':>7} {'Speedup':>8}")
    print(f"  {'-'*14} {'-'*7} {'-'*9} {'-'*6} {'-'*7} {'-'*8}")
    for s in summaries:
        speedup = hf_gen / s["gen_s"]
        audible = "OK" if s["rms"] > 0.005 else "SILENT"
        print(f"  {s['name']:<14} {s['gen_s']:>6.2f}s {s['dur_s']:>7.2f}s {s['rtf']:>6.3f} {s['rms']:>6.4f} {speedup:>6.2f}x  {audible}")
    print(f"{'='*70}")
    print(f"  Audio saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
