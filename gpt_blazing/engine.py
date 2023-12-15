from typing import Optional, Sequence, Tuple, List

import attrs
import torch

from gpt_blazing.model.interface import ModelInference, Role


@attrs.define
class GenerationConfig:
    do_sample: bool = False
    max_new_tokens: int = 2048
    cache_system: bool = True


_default_generation_config = GenerationConfig()


@attrs.define
class Response:
    succeeded: bool
    error_message: str
    content: str
    prompt_tokens: int
    completion_tokens: int


class Engine:

    def __init__(self, model_inference: ModelInference):
        assert model_inference.model_is_ready()
        self.model_inference = model_inference
        self.eos_token = model_inference.get_eos_token()
        self.model_max_length = model_inference.get_model_max_length()

    def greedy_decode(
        self,
        logits: torch.Tensor,
        prompt_tokens: int,
        generation_config: GenerationConfig,
    ):
        sampled_ids: List[int] = []

        input_pos = torch.tensor([prompt_tokens], device=logits.device, dtype=torch.int)
        input_ids = torch.tensor([[0]], device=logits.device, dtype=torch.int)
        for _ in range(
            min(
                generation_config.max_new_tokens,
                self.model_max_length - prompt_tokens,
            )
        ):
            # [1, vocab_size]
            logits = logits[:, -1]
            # Greedy decode.
            # [1, 1]
            token = torch.argmax(logits, dim=1)
            sampled_id = int(token[0])
            if sampled_id == self.eos_token:
                break
            sampled_ids.append(sampled_id)
            # Get next logits.
            input_ids[0][0] = sampled_id
            logits = self.model_inference.model_decode_one_token(
                input_pos=input_pos,
                input_ids=input_ids,
            )
            input_pos += 1

        return Response(
            succeeded=True,
            error_message='',
            content=self.model_inference.tokenizer_decode(sampled_ids),
            prompt_tokens=prompt_tokens,
            completion_tokens=len(sampled_ids),
        )

    def generate(
        self,
        rounds: Sequence[Tuple[Role, str]],
        generation_config: Optional[GenerationConfig] = None,
    ):
        if generation_config is None:
            generation_config = _default_generation_config

        model_prefill_result = self.model_inference.model_prefill(
            rounds=rounds,
            cache_system=generation_config.cache_system,
        )
        if model_prefill_result is None:
            return Response(
                succeeded=False,
                error_message='Failed to prefill model (prompt too long).',
                content='',
                prompt_tokens=-1,
                completion_tokens=-1,
            )

        logits, prompt_tokens = model_prefill_result

        if not generation_config.do_sample:
            return self.greedy_decode(logits, prompt_tokens, generation_config)

        else:
            raise NotImplementedError()
