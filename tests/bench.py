#!/usr/bin/env python3
"""
Final benchmark: PyTorch HF generate vs dual megakernel (backbone + CP).
5 samples each, 3 prompts, wall-clock timing.

Run from repo root: python -m tests.bench
"""
import os
import time
import torch
import numpy as np
import soundfile as sf

from qwentts_turbo.utils import (
    load_model, capture_voice_clone_context, decode_to_audio,
    DEFAULT_MODEL_PATH, DEFAULT_REF_AUDIO, DEFAULT_OUTPUT_DIR,
)
from qwentts_turbo.generator import NativeTTSGenerator

_BUNDLED_REF = os.path.join(os.path.dirname(__file__), "sample_ref.mp3")
REF_AUDIO = DEFAULT_REF_AUDIO or _BUNDLED_REF
REF_TEXT = "This is a reference audio sample for voice cloning"
OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "bench")

PROMPTS = {
    "Short":  "Hello, how are you doing today?",
    "Medium": "Technology is changing the world in ways we never imagined possible.",
    "Long":   "Artificial intelligence is transforming the way we live and work, creating new opportunities and challenges for modern society.",
}

NUM_RUNS = 5
WARMUP_RUNS = 2
MAX_TOKENS = 200
SAMPLE_PARAMS = dict(do_sample=True, temperature=0.9, top_k=50, top_p=1.0)


def capture_for_text(tts, text):
    """Capture voice clone context for a given text."""
    return capture_voice_clone_context(
        tts, text=text, language="English",
        ref_audio_path=REF_AUDIO, ref_text=REF_TEXT,
        max_tokens=MAX_TOKENS,
        x_vector_only_mode=True, **SAMPLE_PARAMS,
    )


def bench_pytorch(tts, text, n_warmup, n_runs):
    """Benchmark HF generate pipeline."""
    ref_audio, ref_sr = sf.read(REF_AUDIO)
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)

    gen_kwargs = dict(text=text, language="English",
                      ref_audio=(ref_audio, ref_sr), ref_text=REF_TEXT,
                      x_vector_only_mode=True,
                      max_new_tokens=MAX_TOKENS, repetition_penalty=1.05, **SAMPLE_PARAMS)

    for _ in range(n_warmup):
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
        results.append({"gen_s": elapsed, "dur_s": dur, "rtf": elapsed / dur,
                        "audio": audio, "sr": sr})
    return results


def bench_mega(gen, captured, n_warmup, n_runs):
    """Benchmark dual megakernel pipeline."""
    ref_code = captured.get('ref_code', [None])[0]
    ie = captured['inputs_embeds']
    am = captured['attention_mask']
    th = captured['trailing_text_hidden']
    tp = captured['tts_pad_embed']
    speech_tok = gen._speech_tokenizer

    for _ in range(n_warmup):
        with torch.no_grad():
            codes = gen.generate(ie, am, th, tp, MAX_TOKENS,
                                 min_new_tokens=20, repetition_penalty=1.2, **SAMPLE_PARAMS)

    results = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            codes = gen.generate(ie, am, th, tp, MAX_TOKENS,
                                 min_new_tokens=20, repetition_penalty=1.2, **SAMPLE_PARAMS)
        audio, sr = decode_to_audio(speech_tok, codes, ref_code)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        dur = len(audio) / sr
        results.append({"gen_s": elapsed, "dur_s": dur, "rtf": elapsed / dur,
                        "audio": audio, "sr": sr})
    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    tts = load_model()

    print("Loading dual megakernel (backbone + CP)...")
    gen = NativeTTSGenerator(model=tts.model,
                             backend="megakernel", cp_backend="megakernel")
    gen._speech_tokenizer = tts.model.speech_tokenizer

    # Capture inputs for each prompt
    print("Capturing voice clone contexts...")
    contexts = {}
    for label, text in PROMPTS.items():
        contexts[label] = capture_for_text(tts, text)

    # Run benchmarks
    all_results = []

    for label, text in PROMPTS.items():
        print(f"\n{'─'*50}")
        print(f"  {label}: {text[:50]}...")
        print(f"{'─'*50}")

        print(f"  PyTorch ({WARMUP_RUNS} warmup + {NUM_RUNS} timed)...")
        pt_runs = bench_pytorch(tts, text, WARMUP_RUNS, NUM_RUNS)
        for i, r in enumerate(pt_runs):
            sf.write(os.path.join(OUTPUT_DIR, f"{label.lower()}_pt_{i}.wav"), r["audio"], r["sr"])
            all_results.append({"prompt": label, "backend": "PyTorch", **{k: r[k] for k in ["gen_s", "dur_s", "rtf"]}})
            print(f"    Run {i+1}: {r['gen_s']:.2f}s → {r['dur_s']:.2f}s audio (RTF {r['rtf']:.3f})")

        print(f"  Mega ({WARMUP_RUNS} warmup + {NUM_RUNS} timed)...")
        mega_runs = bench_mega(gen, contexts[label], WARMUP_RUNS, NUM_RUNS)
        for i, r in enumerate(mega_runs):
            sf.write(os.path.join(OUTPUT_DIR, f"{label.lower()}_mega_{i}.wav"), r["audio"], r["sr"])
            all_results.append({"prompt": label, "backend": "Mega", **{k: r[k] for k in ["gen_s", "dur_s", "rtf"]}})
            print(f"    Run {i+1}: {r['gen_s']:.2f}s → {r['dur_s']:.2f}s audio (RTF {r['rtf']:.3f})")

    # Results table
    print(f"\n┌{'─'*9}┬{'─'*10}┬{'─'*11}┬{'─'*10}┬{'─'*9}┐")
    print(f"│ {'Prompt':<7} │ {'Backend':<8} │ {'Gen Time':<9} │ {'Duration':<8} │ {'RTF':<7} │")
    print(f"├{'─'*9}┼{'─'*10}┼{'─'*11}┼{'─'*10}┼{'─'*9}┤")
    for r in all_results:
        print(f"│ {r['prompt']:<7} │ {r['backend']:<8} │ {r['gen_s']:>7.2f}s │ {r['dur_s']:>6.2f}s │ {r['rtf']:>5.3f}x │")
    print(f"└{'─'*9}┴{'─'*10}┴{'─'*11}┴{'─'*10}┴{'─'*9}┘")

    # Summary per prompt
    print(f"\nSummary:")
    speedups = []
    for label in PROMPTS:
        pt = [r for r in all_results if r["prompt"] == label and r["backend"] == "PyTorch"]
        mg = [r for r in all_results if r["prompt"] == label and r["backend"] == "Mega"]
        pt_mean = np.mean([r["gen_s"] for r in pt])
        mg_mean = np.mean([r["gen_s"] for r in mg])
        pt_dur = np.mean([r["dur_s"] for r in pt])
        mg_dur = np.mean([r["dur_s"] for r in mg])
        sp = pt_mean / mg_mean
        dur_match = abs(pt_dur - mg_dur) / max(pt_dur, 0.01) < 0.5
        speedups.append(sp)
        print(f"  {label:<7}: PT {pt_mean:.2f}s vs Mega {mg_mean:.2f}s → {sp:.2f}x speedup | "
              f"dur PT={pt_dur:.2f}s Mega={mg_dur:.2f}s {'OK' if dur_match else 'MISMATCH'}")

    print(f"\n  Mean speedup: {np.mean(speedups):.2f}x")
    print(f"  Audio saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
