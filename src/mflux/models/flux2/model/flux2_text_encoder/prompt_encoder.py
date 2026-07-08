import time
import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder


class Flux2PromptEncoder:
    @staticmethod
    def encode_prompt(
        prompt: str | list[str],
        tokenizer: Tokenizer,
        text_encoder: Qwen3TextEncoder,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
    ) -> tuple[mx.array, mx.array]:
        """Encodes prompt once to be cached."""
        print(f"🚀 [Text Encoder] Starting text tokenization and hidden layer extraction...")
        start_time = time.perf_counter()
        
        prompt_embeds = Flux2PromptEncoder._get_qwen3_prompt_embeds(
            prompt=prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            max_sequence_length=max_sequence_length,
            hidden_state_layers=text_encoder_out_layers,
        )
        
        # Force MLX to evaluate the text graph immediately so we get an accurate execution time measurement
        mx.eval(prompt_embeds)
        
        text_ids = Flux2PromptEncoder.prepare_text_ids(prompt_embeds)
        mx.eval(text_ids)
        
        end_time = time.perf_counter()
        print(f"✅ [Text Encoder] Successfully encoded prompt features in {(end_time - start_time) * 1000:.2f}ms!")
        
        return prompt_embeds, text_ids

    @staticmethod
    def _get_qwen3_prompt_embeds(
        prompt: str | list[str],
        tokenizer: Tokenizer,
        text_encoder: Qwen3TextEncoder,
        max_sequence_length: int,
        hidden_state_layers: tuple[int, ...],
    ) -> mx.array:
        tokens = tokenizer.tokenize(prompt=prompt, max_length=max_sequence_length)
        return text_encoder.get_prompt_embeds(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            hidden_state_layers=hidden_state_layers,
        )

    @staticmethod
    def prepare_text_ids(x: mx.array, t_coord: mx.array | None = None) -> mx.array:
        batch_size, seq_len, _ = x.shape
        
        if t_coord is None:
            t = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        else:
            t = t_coord.astype(mx.int32)
            if t.ndim == 1:
                t = mx.broadcast_to(t[:, None], (batch_size, seq_len))
            elif t.ndim == 2 and t.shape[1] != seq_len:
                t = mx.broadcast_to(t, (batch_size, seq_len))
            elif t.ndim == 0:
                t = mx.full((batch_size, seq_len), t, dtype=mx.int32)

        h = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        w = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        
        token_ids = mx.arange(seq_len, dtype=mx.int32)[None, :]
        token_ids = mx.broadcast_to(token_ids, (batch_size, seq_len))
        
        return mx.stack([t, h, w, token_ids], axis=-1)
