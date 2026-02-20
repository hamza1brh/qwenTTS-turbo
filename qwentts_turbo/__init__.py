"""qwentts-turbo: CUDA megakernel-accelerated Qwen3-TTS inference."""

from .generator import NativeTTSGenerator
from .utils import load_model, capture_voice_clone_context, decode_to_audio

__version__ = "0.1.0"
