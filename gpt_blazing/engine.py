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
    content: str
    num_tokens: int


class Engine:

    def __init__(self, model_inference: ModelInference):
        self.model_inference = model_inference
        self.eos_token = model_inference.get_eos_token()

    def greedy_decode(
        self,
        logits: torch.Tensor,
        end: int,
        generation_config: GenerationConfig,
    ):
        sampled_ids: List[int] = []

        input_pos = torch.tensor([end], device=logits.device, dtype=torch.int)
        input_ids = torch.tensor([[0]], device=logits.device, dtype=torch.int)
        for _ in range(generation_config.max_new_tokens):
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
            content=self.model_inference.tokenizer_decode(sampled_ids),
            num_tokens=len(sampled_ids),
        )

    def generate(
        self,
        rounds: Sequence[Tuple[Role, str]],
        generation_config: Optional[GenerationConfig] = None,
    ):
        if generation_config is None:
            generation_config = _default_generation_config

        logits, end = self.model_inference.model_prefill(
            rounds=rounds,
            cache_system=generation_config.cache_system,
        )

        if not generation_config.do_sample:
            return self.greedy_decode(logits, end, generation_config)

        else:
            raise NotImplementedError()
