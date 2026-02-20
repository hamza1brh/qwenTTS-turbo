"""
Native TTS Generator: bypasses HuggingFace generate() for the decode loop.

Architecture:
- PyTorch for prefill (fast, parallel)
- Backend choice for backbone decode steps:
  - "pytorch": PyTorch backbone.forward() - accurate, correct EOS
  - "megakernel": CUDA megakernel - faster but may have sampling divergence
- CP backend choice for code predictor:
  - "pytorch": PyTorch CP backbone (100% match with HF)
  - "megakernel": CUDA CP megakernel (4.26x faster, 0.999+ cos_sim)
- torch.mv for codec_head projection
"""
import os
import torch
from transformers.cache_utils import DynamicCache

DEFAULT_MODEL_PATH = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")


class NativeTTSGenerator:
    """Native TTS generation loop, bypassing HF generate() overhead."""

    def __init__(self, model_path=None, model=None, backend="pytorch", cp_backend="pytorch"):
        """
        Args:
            model_path: Path to Qwen3-TTS checkpoint
            model: Optional pre-loaded Qwen3TTSForConditionalGeneration
            backend: "pytorch" (accurate) or "megakernel" (faster backbone decode)
            cp_backend: "pytorch" or "megakernel" (4.26x faster CP decode)
        """
        self.backend = backend
        self.cp_backend = cp_backend
        model_path = model_path or DEFAULT_MODEL_PATH

        if backend == "megakernel":
            from qwentts_turbo.megakernel_backbone import TTSMegakernelGenerator
            print("Loading backbone megakernel...")
            self.backbone_mega = TTSMegakernelGenerator(model_path)
        else:
            self.backbone_mega = None

        if cp_backend == "megakernel":
            from qwentts_turbo.megakernel_code_predictor import CodePredictorMegakernelGenerator
            print("Loading CP megakernel...")
            self.cp_mega = CodePredictorMegakernelGenerator(model_path)
        else:
            self.cp_mega = None

        if model is None:
            from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
            tts = Qwen3TTSModel.from_pretrained(model_path, dtype="bfloat16", device_map="cuda")
            model = tts.model

        talker = model.talker

        # Backbone components
        self.backbone = talker.model  # For prefill
        self.codec_head_weight = talker.codec_head.weight.data  # [3072, 1024]
        self.backbone_codec_embed = talker.model.codec_embedding.weight.data  # [3072, 1024]

        # Code predictor components (PyTorch native, not megakernel)
        self.cp_backbone = talker.code_predictor.model
        self.cp_codec_embeddings = list(self.cp_backbone.codec_embedding)
        self.cp_lm_heads = list(talker.code_predictor.lm_head)

        # Config
        self.eos_token_id = model.config.talker_config.codec_eos_token_id
        self.vocab_size = model.config.talker_config.vocab_size

        # Suppress tokens: top 1024 of vocab except EOS
        self.suppress_mask = torch.zeros(self.vocab_size, device="cuda", dtype=torch.bool)
        for i in range(self.vocab_size - 1024, self.vocab_size):
            if i != self.eos_token_id:
                self.suppress_mask[i] = True

    def generate(self, talker_input_embeds, attention_mask,
                 trailing_text_hidden, tts_pad_embed, max_tokens=500,
                 do_sample=False, temperature=0.9, top_k=50, top_p=1.0,
                 cp_do_sample=False, cp_temperature=0.9, cp_top_k=50,
                 min_new_tokens=2, repetition_penalty=1.0):
        """Run native generation loop.

        Args:
            talker_input_embeds: [1, seq_len, 1024] bf16 prefill embeddings
            attention_mask: [1, seq_len] long
            trailing_text_hidden: [1, num_text_steps, 1024] bf16
            tts_pad_embed: [1, 1, 1024] bf16
            max_tokens: maximum decode steps
            do_sample: use sampling for backbone group0 (vs argmax)
            temperature: sampling temperature for backbone
            top_k: top-k filter for backbone sampling
            top_p: top-p (nucleus) filter for backbone sampling
            cp_do_sample: use sampling for code predictor groups 1-15
            cp_temperature: sampling temperature for code predictor
            cp_top_k: top-k filter for code predictor
            min_new_tokens: suppress EOS for first N steps
            repetition_penalty: penalize repeated g0 tokens (>1.0 to reduce repetition)

        Returns:
            List of [16] codec token lists, one per backbone step.
        """
        # 1. Prefill (PyTorch, fast parallel)
        backbone_hidden, prefill_cache = self._prefill(talker_input_embeds, attention_mask)
        seq_len = talker_input_embeds.shape[1]

        # 2. Setup for decode backend
        if self.backend == "megakernel":
            self._copy_kv_to_megakernel(prefill_cache, seq_len)
        else:
            # PyTorch backend: keep the cache for decode steps
            self._decode_cache = prefill_cache
            self._decode_pos = seq_len

        # 3. Initial group0 from prefill logits
        logits = torch.mv(self.codec_head_weight, backbone_hidden)
        logits[self.suppress_mask] = float('-inf')
        group0 = self._sample_token(
            logits, do_sample, temperature, top_k, top_p,
            self.eos_token_id if min_new_tokens > 0 else None)

        codec_tokens_list = []
        generated_g0 = []  # track g0 history for repetition penalty

        for step in range(max_tokens):
            if group0 == self.eos_token_id:
                break

            generated_g0.append(group0)

            # Code predictor: groups 1-15
            bh = backbone_hidden.unsqueeze(0).unsqueeze(0)
            if bh.dtype != torch.bfloat16:
                bh = bh.to(torch.bfloat16)
            g0_embed = self.backbone_codec_embed[group0].unsqueeze(0).unsqueeze(0)
            groups_1_15 = self._native_cp_loop(
                bh, g0_embed, cp_do_sample, cp_temperature, cp_top_k)

            all_groups = [group0] + groups_1_15
            codec_tokens_list.append(all_groups)

            # Combine 16 codec embeddings
            combined = self._combine_embeddings(all_groups)

            # Add trailing text or pad
            if step < trailing_text_hidden.shape[1]:
                combined = combined + trailing_text_hidden[0, step]
            else:
                combined = combined + tts_pad_embed[0, 0]

            # Backbone decode step
            if self.backend == "megakernel":
                native_hidden = self.backbone_mega.decode_step_with_embedding(combined)
            else:
                native_hidden = self._pytorch_decode_step(combined)

            # Codec head → logits → next group0
            logits = torch.mv(self.codec_head_weight, native_hidden.to(torch.bfloat16))
            logits[self.suppress_mask] = float('-inf')

            # Repetition penalty on g0 history
            if repetition_penalty != 1.0 and generated_g0:
                prev = torch.tensor(generated_g0, device="cuda", dtype=torch.long)
                scores = logits[prev]
                scores = torch.where(scores > 0, scores / repetition_penalty,
                                     scores * repetition_penalty)
                logits[prev] = scores

            # Suppress EOS for early steps
            suppress_eos = self.eos_token_id if step + 1 < min_new_tokens else None
            group0 = self._sample_token(
                logits, do_sample, temperature, top_k, top_p, suppress_eos)

            backbone_hidden = native_hidden

        return codec_tokens_list

    @staticmethod
    def _sample_token(logits, do_sample, temperature, top_k, top_p,
                      suppress_eos_id=None):
        """Sample or argmax from logits."""
        if suppress_eos_id is not None:
            logits = logits.clone()
            logits[suppress_eos_id] = float('-inf')

        if not do_sample:
            return logits.argmax().item()

        scaled = logits / temperature

        # Top-k filtering
        if top_k > 0:
            topk_vals = torch.topk(scaled, min(top_k, scaled.size(-1)))[0]
            scaled[scaled < topk_vals[-1]] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            sorted_logits[remove] = float('-inf')
            scaled.scatter_(0, sorted_idx, sorted_logits)

        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, 1).item()

    def _prefill(self, inputs_embeds, attention_mask):
        """Run PyTorch prefill on backbone (fast, parallel)."""
        cache = DynamicCache()
        seq_len = inputs_embeds.shape[1]
        pos_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).unsqueeze(0).expand(3, 1, -1)

        with torch.no_grad():
            out = self.backbone(
                inputs_embeds=inputs_embeds,
                position_ids=pos_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )

        hidden = out.last_hidden_state[0, -1, :]  # [1024]
        return hidden, out.past_key_values

    def _pytorch_decode_step(self, combined):
        """Run a single decode step using PyTorch backbone."""
        decode_embeds = combined.unsqueeze(0).unsqueeze(0)
        decode_pos = torch.tensor([[[self._decode_pos]]], device="cuda").expand(3, 1, 1)
        decode_mask = torch.ones(1, self._decode_pos + 1, device="cuda", dtype=torch.long)

        with torch.no_grad():
            out = self.backbone(
                inputs_embeds=decode_embeds,
                position_ids=decode_pos,
                attention_mask=decode_mask,
                past_key_values=self._decode_cache,
                use_cache=True,
            )
        self._decode_cache = out.past_key_values
        self._decode_pos += 1
        return out.last_hidden_state[0, -1, :]

    def _copy_kv_to_megakernel(self, cache, seq_len):
        """Copy PyTorch DynamicCache to megakernel KV cache."""
        self.backbone_mega.reset()
        k_list, v_list = [], []
        for layer_idx in range(28):
            k, v = cache[layer_idx]
            k_list.append(k.squeeze(0))
            v_list.append(v.squeeze(0))
        self.backbone_mega.set_kv_cache(
            torch.stack(k_list, dim=0),
            torch.stack(v_list, dim=0),
        )
        self.backbone_mega.set_position(seq_len)

    def _native_cp_loop(self, backbone_hidden, codec0_embed,
                        do_sample=False, temperature=0.9, top_k=50):
        """Native CP loop - PyTorch or megakernel."""
        if self.cp_backend == "megakernel":
            return self._native_cp_loop_megakernel(
                backbone_hidden, codec0_embed, do_sample, temperature, top_k)
        else:
            return self._native_cp_loop_pytorch(
                backbone_hidden, codec0_embed, do_sample, temperature, top_k)

    def _native_cp_loop_pytorch(self, backbone_hidden, codec0_embed,
                                do_sample=False, temperature=0.9, top_k=50):
        """Native CP loop using PyTorch backbone (Phase 5: 100% match with HF)."""
        cp_input = torch.cat([backbone_hidden, codec0_embed], dim=1)
        cache = DynamicCache()

        with torch.no_grad():
            out = self.cp_backbone(
                inputs_embeds=cp_input,
                position_ids=torch.arange(2, device="cuda").unsqueeze(0),
                attention_mask=torch.ones(1, 2, device="cuda", dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            )
        cache = out.past_key_values
        hidden = out.last_hidden_state[0, -1, :]
        logits = torch.mv(self.cp_lm_heads[0].weight.data, hidden)
        tokens = [self._sample_token(logits, do_sample, temperature, top_k, 1.0)]

        for g in range(1, 15):
            token_t = torch.tensor([[tokens[-1]]], device="cuda")
            embed = self.cp_codec_embeddings[g - 1](token_t)
            pos = torch.tensor([[g + 1]], device="cuda")
            mask = torch.ones(1, g + 2, device="cuda", dtype=torch.long)

            with torch.no_grad():
                out = self.cp_backbone(
                    inputs_embeds=embed,
                    position_ids=pos,
                    attention_mask=mask,
                    past_key_values=cache,
                    use_cache=True,
                )
            cache = out.past_key_values
            hidden = out.last_hidden_state[0, -1, :]
            logits = torch.mv(self.cp_lm_heads[g].weight.data, hidden)
            tokens.append(self._sample_token(logits, do_sample, temperature, top_k, 1.0))

        return tokens

    def _native_cp_loop_megakernel(self, backbone_hidden, codec0_embed,
                                   do_sample=False, temperature=0.9, top_k=50):
        """Native CP loop using megakernel (Phase 9: 4.26x faster, 0.999+ cos_sim)."""
        cp_input = torch.cat([backbone_hidden, codec0_embed], dim=1)

        # Reset megakernel and do PyTorch prefill (same as PyTorch version)
        self.cp_mega.reset()
        cache = DynamicCache()

        with torch.no_grad():
            out = self.cp_backbone(
                inputs_embeds=cp_input,
                position_ids=torch.arange(2, device="cuda").unsqueeze(0),
                attention_mask=torch.ones(1, 2, device="cuda", dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            )

        prefill_hidden = out.last_hidden_state[0, -1, :]
        prefill_cache = out.past_key_values

        # Group 1 from prefill
        logits = torch.mv(self.cp_lm_heads[0].weight.data, prefill_hidden)
        tokens = [self._sample_token(logits, do_sample, temperature, top_k, 1.0)]

        # Copy prefill KV cache to megakernel
        k_list, v_list = [], []
        for layer_idx in range(5):
            k, v = prefill_cache[layer_idx]
            k_list.append(k.squeeze(0))
            v_list.append(v.squeeze(0))
        self.cp_mega.decoder.set_kv_cache(
            torch.stack(k_list, dim=0),
            torch.stack(v_list, dim=0),
        )
        self.cp_mega.decoder.set_position(2)

        # Decode 14 steps via megakernel
        for g in range(1, 15):
            token_t = torch.tensor([tokens[-1]], device="cuda")
            embed = self.cp_codec_embeddings[g - 1](token_t).squeeze(0)  # [1024]
            hidden = self.cp_mega.decode_step_with_embedding(embed)  # fp32 [1024]
            logits = torch.mv(self.cp_lm_heads[g].weight.data, hidden.to(torch.bfloat16))
            tokens.append(self._sample_token(logits, do_sample, temperature, top_k, 1.0))

        return tokens

    def _combine_embeddings(self, groups):
        """Sum of 16 codec embeddings (backbone group0 + 15 CP groups)."""
        combined = self.backbone_codec_embed[groups[0]].clone()
        for i in range(15):
            token_t = torch.tensor([groups[i + 1]], device="cuda")
            combined = combined + self.cp_codec_embeddings[i](token_t).squeeze()
        return combined
