from typing import Optional, Sequence, Tuple, List, Callable, TypeVar
import logging

import attrs
import torch

from gpt_blazing.model.interface import ModelInference, Role

logger = logging.getLogger(__name__)


@attrs.define
class GenerationConfig:
    # Control length.
    max_new_tokens: int = 2048
    # Control strategy.
    do_sample: bool = False
    # Control logits.
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 1.0
    contrastive_penalty_alpha: Optional[float] = 0.6
    # Control cache.
    cache_system: bool = True


_default_generation_config = GenerationConfig()


@attrs.define
class Response:
    succeeded: bool
    error_message: str
    content: str
    num_prompt_tokens: int
    num_completion_tokens: int


def torch_strategy_greedy_sample_from_logits(logits: torch.Tensor):
    logits = logits[0, -1]
    sampled_id = int(torch.argmax(logits))
    return sampled_id


def torch_strategy_sample_sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
):
    # [vocab_size]
    logits = logits[0, -1]

    # Apply temperature.
    logits /= temperature

    # Apply top_k.
    assert top_k > 0
    top_k = min(top_k, logits.size(-1))
    top_k_values, top_k_indices = torch.topk(logits, top_k, sorted=False)

    # To probs.
    top_k_probs = torch.softmax(top_k_values, dim=-1)

    # Sample a token.
    sampled_idx_in_top_k = torch.multinomial(top_k_probs, num_samples=1)[0]
    sampled_id = top_k_indices[sampled_idx_in_top_k].to(dtype=torch.int)
    return int(sampled_id)


_T_FUNC = TypeVar('_T_FUNC')


def torch_compile_func_sample_from_logits(func: _T_FUNC) -> _T_FUNC:
    # fullgraph can not be set here.
    return torch.compile(func, dynamic=True)  # type: ignore


class Engine:

    def __init__(
        self,
        model_inference: ModelInference,
        skip_torch_compile: bool = False,
    ):
        assert model_inference.model_is_ready()
        self.model_inference = model_inference
        self.eos_token = model_inference.get_eos_token()
        self.model_max_length = model_inference.get_model_max_length()

        if not skip_torch_compile:
            logger.info('Compiling engine opts...')
            self.torch_strategy_greedy_sample_from_logits = \
                torch_compile_func_sample_from_logits(torch_strategy_greedy_sample_from_logits)

            self.torch_strategy_sample_sample_from_logits = \
                torch_compile_func_sample_from_logits(torch_strategy_sample_sample_from_logits)

            for warmup_generation_config in [
                GenerationConfig(do_sample=True),
                GenerationConfig(do_sample=False),
            ]:
                logger.info(f'warmup_generation_config={warmup_generation_config}')
                for _ in range(3):
                    logger.info(
                        str(
                            self.generate(
                                [(Role.USER, '你好')],
                                generation_config=warmup_generation_config,
                            )
                        )
                    )

        else:
            logger.info('skip_torch_compile is set, abort. (only for debugging)')
            self.torch_strategy_greedy_sample_from_logits = \
                torch_strategy_greedy_sample_from_logits

            self.torch_strategy_sample_sample_from_logits = \
                torch_strategy_sample_sample_from_logits

    def get_current_max_new_tokens(
        self,
        num_prompt_tokens: int,
        generation_config: GenerationConfig,
    ):
        return min(
            generation_config.max_new_tokens,
            self.model_max_length - num_prompt_tokens,
        )

    def strategy_procedure_sample_one_token(
        self,
        func_sample_from_logits: Callable[[torch.Tensor, GenerationConfig], int],
        logits: torch.Tensor,
        num_prompt_tokens: int,
        generation_config: GenerationConfig,
    ):
        sampled_ids: List[int] = []

        input_pos = torch.tensor([num_prompt_tokens], device=logits.device, dtype=torch.int)
        input_ids = torch.tensor([[0]], device=logits.device, dtype=torch.int)
        for _ in range(self.get_current_max_new_tokens(num_prompt_tokens, generation_config)):
            with torch.inference_mode():
                sampled_id = func_sample_from_logits(logits, generation_config)
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
            num_prompt_tokens=num_prompt_tokens,
            num_completion_tokens=len(sampled_ids),
        )

    def strategy_greedy_sample_from_logits(
        self,
        logits: torch.Tensor,
        generation_config: GenerationConfig,
    ):
        return self.torch_strategy_greedy_sample_from_logits(logits)

    def strategy_greedy(
        self,
        logits: torch.Tensor,
        num_prompt_tokens: int,
        generation_config: GenerationConfig,
    ):
        return self.strategy_procedure_sample_one_token(
            func_sample_from_logits=self.strategy_greedy_sample_from_logits,
            logits=logits,
            num_prompt_tokens=num_prompt_tokens,
            generation_config=generation_config,
        )

    def strategy_contrastive(
        self,
        logits: torch.Tensor,
        num_prompt_tokens: int,
        generation_config: GenerationConfig,
    ):
        '''
        TODO
        Su and Collier, 2022 "Contrastive Search Is What You Need For Neural Text Generation"
        https://huggingface.co/blog/introducing-csearch
        https://github.com/huggingface/transformers/blob/238d2e3c44366aba9dc5c770c95475765a6725cb/src/transformers/generation/utils.py#L1968
        ''' # noqa

    def strategy_sample_sample_from_logits(
        self,
        logits: torch.Tensor,
        generation_config: GenerationConfig,
    ):
        return self.torch_strategy_sample_sample_from_logits(
            logits=logits,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
        )

    def strategy_sample(
        self,
        logits: torch.Tensor,
        num_prompt_tokens: int,
        generation_config: GenerationConfig,
    ):
        return self.strategy_procedure_sample_one_token(
            func_sample_from_logits=self.strategy_sample_sample_from_logits,
            logits=logits,
            num_prompt_tokens=num_prompt_tokens,
            generation_config=generation_config,
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
                num_prompt_tokens=-1,
                num_completion_tokens=-1,
            )

        logits, num_prompt_tokens = model_prefill_result

        if not generation_config.do_sample:
            func_strategy = self.strategy_greedy
        else:
            func_strategy = self.strategy_sample

        return func_strategy(logits, num_prompt_tokens, generation_config)
