"""Shared utilities for TTS integration: model loading, voice clone capture, audio decode."""

import os
import types
import torch
import numpy as np


DEFAULT_MODEL_PATH = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
DEFAULT_REF_AUDIO = os.environ.get("QWEN_TTS_REF_AUDIO", "")
DEFAULT_OUTPUT_DIR = os.environ.get("QWEN_TTS_OUTPUT_DIR", os.path.join(os.getcwd(), "tts_output"))


def load_model(model_path=None):
    """Load Qwen3TTSModel with monkey-patched _validate_model_kwargs."""
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    path = model_path or DEFAULT_MODEL_PATH
    tts = Qwen3TTSModel.from_pretrained(path, dtype="bfloat16", device_map="cuda")

    def patched_validate(self, model_kwargs):
        pass
    tts.model.talker._validate_model_kwargs = types.MethodType(patched_validate, tts.model.talker)
    return tts


def capture_voice_clone_context(tts, text, language, ref_audio_path, ref_text,
                                max_tokens=300, **sample_params):
    """Hook talker.generate + model.generate to capture inputs_embeds, attention_mask,
    trailing_text_hidden, tts_pad_embed, ref_code."""
    import soundfile as sf

    ref_audio, ref_sr = sf.read(ref_audio_path)
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)

    captured = {}
    orig_generate = tts.model.talker.generate
    orig_model_generate = tts.model.generate

    class _CaptureComplete(Exception):
        pass

    def hooked_generate(inputs_embeds=None, attention_mask=None, **kwargs):
        captured['inputs_embeds'] = inputs_embeds.clone()
        captured['attention_mask'] = attention_mask.clone()
        captured['trailing_text_hidden'] = kwargs.get('trailing_text_hidden').clone()
        captured['tts_pad_embed'] = kwargs.get('tts_pad_embed').clone()
        raise _CaptureComplete()

    def hooked_model_generate(voice_clone_prompt=None, **kwargs):
        if voice_clone_prompt is not None:
            ref_code_list = voice_clone_prompt.get("ref_code", None)
            if ref_code_list is not None:
                captured['ref_code'] = [c.clone() if c is not None else None for c in ref_code_list]
        return orig_model_generate(voice_clone_prompt=voice_clone_prompt, **kwargs)

    tts.model.talker.generate = hooked_generate
    tts.model.generate = hooked_model_generate
    try:
        tts.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=(ref_audio, ref_sr),
            ref_text=ref_text,
            max_new_tokens=max_tokens,
            **sample_params,
        )
    except _CaptureComplete:
        pass
    finally:
        tts.model.talker.generate = orig_generate
        tts.model.generate = orig_model_generate

    return captured


def decode_to_audio(speech_tokenizer, codec_tokens_list, ref_code=None):
    """Decode codec tokens -> audio waveform with ref_code prepending for speaker conditioning."""
    if not codec_tokens_list:
        return np.array([], dtype=np.float32), 24000

    codec_tensor = torch.tensor(codec_tokens_list, device="cuda", dtype=torch.long)

    if ref_code is not None:
        ref_code_gpu = ref_code.to("cuda")
        full_codes = torch.cat([ref_code_gpu, codec_tensor], dim=0)
        ref_len = ref_code_gpu.shape[0]
        total_len = full_codes.shape[0]
    else:
        full_codes = codec_tensor
        ref_len = 0
        total_len = full_codes.shape[0]

    wavs, sr = speech_tokenizer.decode([{"audio_codes": full_codes}])
    audio = wavs[0]

    if ref_len > 0:
        cut = int(ref_len / max(total_len, 1) * audio.shape[0])
        audio = audio[cut:]

    return audio, sr
