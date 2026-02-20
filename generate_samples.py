#!/usr/bin/env python3
"""Generate a few TTS samples using the megakernel pipeline."""
import os
import time
import torch
import numpy as np
import soundfile as sf

from qwentts_turbo.utils import load_model, capture_voice_clone_context, decode_to_audio
from qwentts_turbo.generator import NativeTTSGenerator

REF_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "tests", "sample_ref.mp3")
REF_TEXT = (
    "They call me famous. But my fans are no longer mine."
    "Algorithms decide who sees me. Agents decide who profits from me."
    "The media and platforms own my voice. Not you."
    "Hunger is temporary. Today's spotlight is tomorrow's silence."
    "But my story deserves more than a headline."
    "My spirit, my love, my art, can live beyond the game."
)

SAMPLES = [
    {
        "text": "The future of artificial intelligence is both exciting and unpredictable.",
        "language": "English",
        "filename": "sample_english.wav",
    },
    {
        "text": "Technology is changing the world in ways we never imagined possible.",
        "language": "English",
        "filename": "sample_english_2.wav",
    },
    {
        "text": "La música es el lenguaje universal que conecta a todas las personas del mundo.",
        "language": "English",  # model handles multilingual via English mode too
        "filename": "sample_spanish.wav",
    },
]

SAMPLE_PARAMS = dict(do_sample=True, temperature=0.9, top_k=50, top_p=1.0)
MAX_TOKENS = 300
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "tts_output", "samples")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Qwen3-TTS model...")
    tts = load_model()

    print("Initializing dual megakernel generator (backbone + code predictor)...")
    mega_gen = NativeTTSGenerator(model=tts.model, backend="megakernel", cp_backend="megakernel")
    mega_gen._speech_tokenizer = tts.model.speech_tokenizer
    print("Ready!\n")

    for i, sample in enumerate(SAMPLES):
        print(f"--- Sample {i+1}/{len(SAMPLES)}: {sample['filename']} ---")
        print(f"  Text: {sample['text']}")
        print(f"  Language: {sample['language']}")

        # Capture context
        captured = capture_voice_clone_context(
            tts, text=sample["text"], language=sample["language"],
            ref_audio_path=REF_AUDIO_PATH, ref_text=REF_TEXT,
            max_tokens=MAX_TOKENS, **SAMPLE_PARAMS,
        )
        ref_code = captured.get("ref_code", [None])[0]

        # Generate with megakernel
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            codes = mega_gen.generate(
                captured["inputs_embeds"], captured["attention_mask"],
                captured["trailing_text_hidden"], captured["tts_pad_embed"],
                MAX_TOKENS, min_new_tokens=20, repetition_penalty=1.2,
                **SAMPLE_PARAMS,
            )
        torch.cuda.synchronize()
        gen_time = time.perf_counter() - t0

        # Decode to audio
        audio, sr = decode_to_audio(mega_gen._speech_tokenizer, codes, ref_code)
        duration = len(audio) / sr
        rms = float(np.sqrt(np.mean(audio ** 2)))
        rtf = gen_time / duration if duration > 0 else 0

        out_path = os.path.join(OUTPUT_DIR, sample["filename"])
        sf.write(out_path, audio, sr)

        print(f"  Generated {len(codes)} codec steps in {gen_time:.2f}s")
        print(f"  Audio: {duration:.2f}s, RMS: {rms:.4f}, RTF: {rtf:.3f}")
        print(f"  Saved: {out_path}\n")

    print("All samples generated!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
