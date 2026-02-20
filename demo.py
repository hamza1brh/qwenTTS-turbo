"""Gradio demo for qwentts-turbo — megakernel-accelerated voice synthesis."""
import os
import time
import tempfile
import torch
import numpy as np
import soundfile as sf
import gradio as gr

# Global state — loaded once at startup
_tts = None
_mega_gen = None

SAMPLE_PARAMS = dict(do_sample=True, temperature=0.9, top_k=50, top_p=1.0)


def _load_models():
    """Load TTS model and megakernel generator (cached globally)."""
    global _tts, _mega_gen
    if _tts is not None:
        return

    from qwentts_turbo.utils import load_model
    from qwentts_turbo.generator import NativeTTSGenerator

    print("Loading Qwen3-TTS model...")
    _tts = load_model()

    print("Loading dual megakernel (backbone + CP)...")
    _mega_gen = NativeTTSGenerator(model=_tts.model,
                                   backend="megakernel", cp_backend="megakernel")
    _mega_gen._speech_tokenizer = _tts.model.speech_tokenizer
    print("Ready!")


def _capture_context(text, language, ref_audio_path, ref_text, max_tokens):
    """Capture voice clone context for megakernel pipeline."""
    from qwentts_turbo.utils import capture_voice_clone_context
    return capture_voice_clone_context(
        _tts, text=text, language=language,
        ref_audio_path=ref_audio_path, ref_text=ref_text,
        max_tokens=max_tokens, **SAMPLE_PARAMS,
    )


def _generate_pytorch(text, language, ref_audio_path, ref_text, max_tokens):
    """Generate audio using PyTorch HF pipeline."""
    ref_audio, ref_sr = sf.read(ref_audio_path)
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wavs, sr = _tts.generate_voice_clone(
        text=text, language=language,
        ref_audio=(ref_audio, ref_sr), ref_text=ref_text,
        max_new_tokens=max_tokens, repetition_penalty=1.05,
        **SAMPLE_PARAMS,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    audio = wavs[0]
    duration = len(audio) / sr
    return audio, sr, elapsed, duration


def _generate_megakernel(text, language, ref_audio_path, ref_text, max_tokens):
    """Generate audio using dual megakernel pipeline."""
    from qwentts_turbo.utils import decode_to_audio

    captured = _capture_context(text, language, ref_audio_path, ref_text, max_tokens)
    ref_code = captured.get('ref_code', [None])[0]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        codes = _mega_gen.generate(
            captured['inputs_embeds'], captured['attention_mask'],
            captured['trailing_text_hidden'], captured['tts_pad_embed'],
            max_tokens, min_new_tokens=20, repetition_penalty=1.2,
            **SAMPLE_PARAMS,
        )
    audio, sr = decode_to_audio(_mega_gen._speech_tokenizer, codes, ref_code)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    duration = len(audio) / sr
    return audio, sr, elapsed, duration


def _save_temp_wav(audio, sr):
    """Save audio to a temporary WAV file for Gradio."""
    path = tempfile.mktemp(suffix=".wav")
    sf.write(path, audio, sr)
    return path


def generate(text, language, ref_audio, ref_text, backend, max_tokens):
    """Main generation function called by Gradio."""
    _load_models()

    if not text.strip():
        return None, None, "Please enter text to synthesize."
    if ref_audio is None:
        return None, None, "Please upload a reference audio file."

    max_tokens = int(max_tokens)

    if backend == "Both (comparison)":
        # PyTorch
        pt_audio, pt_sr, pt_time, pt_dur = _generate_pytorch(
            text, language, ref_audio, ref_text, max_tokens)
        pt_rtf = pt_time / pt_dur if pt_dur > 0 else 0
        pt_path = _save_temp_wav(pt_audio, pt_sr)

        # Megakernel
        mk_audio, mk_sr, mk_time, mk_dur = _generate_megakernel(
            text, language, ref_audio, ref_text, max_tokens)
        mk_rtf = mk_time / mk_dur if mk_dur > 0 else 0
        mk_path = _save_temp_wav(mk_audio, mk_sr)

        speedup = pt_time / mk_time if mk_time > 0 else 0
        stats = (
            f"PyTorch: {pt_time:.2f}s gen -> {pt_dur:.2f}s audio (RTF {pt_rtf:.3f})\n"
            f"Megakernel: {mk_time:.2f}s gen -> {mk_dur:.2f}s audio (RTF {mk_rtf:.3f})\n"
            f"Speedup: {speedup:.2f}x"
        )
        return pt_path, mk_path, stats

    elif backend == "Megakernel":
        audio, sr, elapsed, duration = _generate_megakernel(
            text, language, ref_audio, ref_text, max_tokens)
        rtf = elapsed / duration if duration > 0 else 0
        path = _save_temp_wav(audio, sr)
        stats = f"Megakernel: {elapsed:.2f}s gen -> {duration:.2f}s audio (RTF {rtf:.3f})"
        return path, None, stats

    else:  # PyTorch
        audio, sr, elapsed, duration = _generate_pytorch(
            text, language, ref_audio, ref_text, max_tokens)
        rtf = elapsed / duration if duration > 0 else 0
        path = _save_temp_wav(audio, sr)
        stats = f"PyTorch: {elapsed:.2f}s gen -> {duration:.2f}s audio (RTF {rtf:.3f})"
        return path, None, stats


def build_ui():
    with gr.Blocks(title="qwentts-turbo") as app:
        gr.Markdown("# qwentts-turbo\nCUDA megakernel-accelerated voice synthesis for Qwen3-TTS")

        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text to synthesize", lines=3,
                    value="Technology is changing the world in ways we never imagined possible.")
                language = gr.Dropdown(
                    ["English", "Chinese", "Japanese", "Korean"],
                    value="English", label="Language")
                ref_audio = gr.Audio(label="Reference audio (for voice cloning)",
                                     type="filepath")
                ref_text = gr.Textbox(label="Reference text (transcript of ref audio)", lines=2,
                    value="This is a reference audio sample for voice cloning")
                backend = gr.Radio(
                    ["Megakernel", "PyTorch", "Both (comparison)"],
                    value="Both (comparison)", label="Backend")
                max_tokens = gr.Slider(50, 500, value=200, step=10, label="Max tokens")
                btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                audio1 = gr.Audio(label="Output (PyTorch / primary)", type="filepath")
                audio2 = gr.Audio(label="Output (Megakernel / comparison)", type="filepath")
                stats = gr.Textbox(label="Stats", lines=4)

        btn.click(generate,
                  inputs=[text, language, ref_audio, ref_text, backend, max_tokens],
                  outputs=[audio1, audio2, stats])

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(share=False)
