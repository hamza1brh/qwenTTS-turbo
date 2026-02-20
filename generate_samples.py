#!/usr/bin/env python3
"""Generate diverse TTS samples using the dual megakernel pipeline."""
import os
import time
import torch
import numpy as np
import soundfile as sf

from qwentts_turbo.utils import load_model, capture_voice_clone_context, decode_to_audio
from qwentts_turbo.generator import NativeTTSGenerator

REF_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "tests", "sample_ref.mp3")
REF_TEXT = (
    "They call me famous. But my fans are no longer mine. "
    "Algorithms decide who sees me. Agents decide who profits from me. "
    "The media and platforms own my voice. Not you. "
    "Hunger is temporary. Today's spotlight is tomorrow's silence. "
    "But my story deserves more than a headline. "
    "My spirit, my love, my art, can live beyond the game."
)

SAMPLES = [
    # --- Short (1-2 sentences) ---
    {
        "text": "Hello, welcome to the future of speech synthesis.",
        "language": "English",
        "max_tokens": 150,
        "filename": "en_short.wav",
    },
    {
        "text": "Hola, bienvenido al futuro de la síntesis de voz.",
        "language": "English",
        "max_tokens": 150,
        "filename": "es_short.wav",
    },
    {
        "text": "你好，欢迎来到语音合成的未来。",
        "language": "Chinese",
        "max_tokens": 150,
        "filename": "zh_short.wav",
    },
    {
        "text": "こんにちは、音声合成の未来へようこそ。",
        "language": "Japanese",
        "max_tokens": 150,
        "filename": "ja_short.wav",
    },
    {
        "text": "안녕하세요, 음성 합성의 미래에 오신 것을 환영합니다.",
        "language": "Korean",
        "max_tokens": 150,
        "filename": "ko_short.wav",
    },
    # --- Medium (2-3 sentences) ---
    {
        "text": (
            "The megakernel fuses an entire transformer pass into a single CUDA launch. "
            "This eliminates kernel dispatch overhead and achieves a three point seven five x speedup."
        ),
        "language": "English",
        "max_tokens": 300,
        "filename": "en_medium.wav",
    },
    {
        "text": (
            "Die Technologie verändert die Welt auf eine Weise, die wir uns nie vorstellen konnten. "
            "Jeden Tag bringen neue Entdeckungen Veränderungen in unser Leben und unsere Arbeit."
        ),
        "language": "English",
        "max_tokens": 300,
        "filename": "de_medium.wav",
    },
    {
        "text": (
            "La inteligencia artificial está transformando todas las industrias del mundo. "
            "Desde la medicina hasta la educación, su impacto es profundo e irreversible."
        ),
        "language": "English",
        "max_tokens": 300,
        "filename": "es_medium.wav",
    },
    {
        "text": (
            "人工智能正在改变我们生活的方方面面。从医疗到教育，"
            "从交通到娱乐，智能技术无处不在。"
        ),
        "language": "Chinese",
        "max_tokens": 300,
        "filename": "zh_medium.wav",
    },
    # --- Long (paragraph) ---
    {
        "text": (
            "Throughout history, the greatest breakthroughs in science and technology "
            "have come from those who dared to challenge conventional wisdom. "
            "From Galileo's telescope to the invention of the transistor, "
            "every major leap forward required courage, persistence, and an unwavering "
            "belief that the impossible could become possible."
        ),
        "language": "English",
        "max_tokens": 500,
        "filename": "en_long.wav",
    },
    {
        "text": (
            "A lo largo de la historia, los mayores avances en ciencia y tecnología "
            "han venido de aquellos que se atrevieron a desafiar el pensamiento convencional. "
            "Desde el telescopio de Galileo hasta la invención del transistor, cada gran salto "
            "adelante requirió coraje, persistencia y una creencia inquebrantable "
            "de que lo imposible podría hacerse posible."
        ),
        "language": "English",
        "max_tokens": 500,
        "filename": "es_long.wav",
    },
    {
        "text": (
            "在人类历史的长河中，科学与技术的每一次重大突破，"
            "都来自那些敢于挑战传统观念的先驱者。"
            "从伽利略的望远镜到晶体管的发明，每一次飞跃都需要勇气、"
            "坚持和对不可能变为可能的坚定信念。"
        ),
        "language": "Chinese",
        "max_tokens": 500,
        "filename": "zh_long.wav",
    },
]

SAMPLE_PARAMS = dict(do_sample=True, temperature=0.9, top_k=50, top_p=1.0)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "tts_output")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Qwen3-TTS model...")
    tts = load_model()

    print("Initializing dual megakernel generator (backbone + code predictor)...")
    mega_gen = NativeTTSGenerator(model=tts.model, backend="megakernel", cp_backend="megakernel")
    mega_gen._speech_tokenizer = tts.model.speech_tokenizer
    print("Ready!\n")

    total_audio = 0.0
    total_gen = 0.0

    for i, sample in enumerate(SAMPLES):
        max_tokens = sample["max_tokens"]
        print(f"--- [{i+1}/{len(SAMPLES)}] {sample['filename']} ({sample['language']}, max_tokens={max_tokens}) ---")
        print(f"  Text: {sample['text'][:80]}{'...' if len(sample['text']) > 80 else ''}")

        captured = capture_voice_clone_context(
            tts, text=sample["text"], language=sample["language"],
            ref_audio_path=REF_AUDIO_PATH, ref_text=REF_TEXT,
            max_tokens=max_tokens, **SAMPLE_PARAMS,
        )
        ref_code = captured.get("ref_code", [None])[0]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            codes = mega_gen.generate(
                captured["inputs_embeds"], captured["attention_mask"],
                captured["trailing_text_hidden"], captured["tts_pad_embed"],
                max_tokens, min_new_tokens=20, repetition_penalty=1.2,
                **SAMPLE_PARAMS,
            )
        torch.cuda.synchronize()
        gen_time = time.perf_counter() - t0

        audio, sr = decode_to_audio(mega_gen._speech_tokenizer, codes, ref_code)
        duration = len(audio) / sr
        rms = float(np.sqrt(np.mean(audio ** 2)))
        rtf = gen_time / duration if duration > 0 else 0

        out_path = os.path.join(OUTPUT_DIR, sample["filename"])
        sf.write(out_path, audio, sr)

        total_audio += duration
        total_gen += gen_time

        print(f"  {len(codes)} steps | {gen_time:.2f}s gen | {duration:.2f}s audio | RTF {rtf:.3f} | RMS {rms:.4f}")
        print(f"  -> {out_path}\n")

    print("=" * 60)
    print(f"Generated {len(SAMPLES)} samples")
    print(f"Total audio: {total_audio:.1f}s | Total gen time: {total_gen:.1f}s | Avg RTF: {total_gen/total_audio:.3f}")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
