from typing import Optional, Sequence, Tuple, List, Callable, TypeVar, Set
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
    top_p: float = 1.0
    top_k: int = 6
    contrastive_penalty_alpha: float = 0.0
    contrastive_context_size: int = 20
    contrastive_similarity_thr: float = 0.95
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


def torch_compile_opt(func: _T_FUNC) -> _T_FUNC:
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
        self.hidden_size = model_inference.get_hidden_size()

        # For contrastive search.
        self.contrastive_prev_norm_hidden_states: Optional[torch.Tensor] = None

        if not skip_torch_compile:
            logger.info('Compiling engine opts...')
            self.torch_strategy_greedy_sample_from_logits = \
                torch_compile_opt(torch_strategy_greedy_sample_from_logits)

            self.torch_strategy_sample_sample_from_logits = \
                torch_compile_opt(torch_strategy_sample_sample_from_logits)

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

    def strategy_procedure_sample_token_based_on_current_logits(
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
            sampled_id = func_sample_from_logits(logits, generation_config)
            if sampled_id == self.eos_token:
                break
            sampled_ids.append(sampled_id)
            # Get next logits.
            input_ids[0][0] = sampled_id
            logits, _ = self.model_inference.model_decode_one_token(
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
        hidden_states: torch.Tensor,
        num_prompt_tokens: int,
        generation_config: GenerationConfig,
    ):
        return self.strategy_procedure_sample_token_based_on_current_logits(
            func_sample_from_logits=self.strategy_greedy_sample_from_logits,
            logits=logits,
            num_prompt_tokens=num_prompt_tokens,
            generation_config=generation_config,
        )

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
        hidden_states: torch.Tensor,
        num_prompt_tokens: int,
        generation_config: GenerationConfig,
    ):
        return self.strategy_procedure_sample_token_based_on_current_logits(
            func_sample_from_logits=self.strategy_sample_sample_from_logits,
            logits=logits,
            num_prompt_tokens=num_prompt_tokens,
            generation_config=generation_config,
        )

    def prepare_contrastive_search(
        self,
        hidden_states: torch.Tensor,
        generation_config: GenerationConfig,
    ):
        context_size = generation_config.contrastive_context_size
        if (
            self.contrastive_prev_norm_hidden_states is not None
            and self.contrastive_prev_norm_hidden_states.size(0) >= context_size
        ):
            return

        self.contrastive_prev_norm_hidden_states = torch.empty(
            (context_size, self.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    def strategy_contrastive(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        num_prompt_tokens: int,
        generation_config: GenerationConfig,
    ):
        self.prepare_contrastive_search(hidden_states, generation_config)

        prev_norm_hidden_states = self.contrastive_prev_norm_hidden_states
        assert prev_norm_hidden_states is not None

        # [1, hidden_size]
        hidden_states = hidden_states[0, -1].unsqueeze(0)
        hidden_states /= hidden_states.norm()
        prev_norm_hidden_states[0] = hidden_states
        del hidden_states
        prev_norm_hidden_states_pos = 1

        top_k = generation_config.top_k
        assert top_k > 0
        top_k = min(top_k, logits.size(-1))

        penalty_alpha = generation_config.contrastive_penalty_alpha
        assert penalty_alpha > 0

        similarity_thr = generation_config.contrastive_similarity_thr

        sampled_ids: List[int] = []

        input_pos = torch.tensor([num_prompt_tokens], device=logits.device, dtype=torch.int)
        input_ids = torch.tensor([[0]], device=logits.device, dtype=torch.int)

        for _ in range(self.get_current_max_new_tokens(num_prompt_tokens, generation_config)):
            logits = logits[0, -1]

            top_k_values, top_k_indices = torch.topk(logits, top_k)
            top_k_probs = torch.softmax(top_k_values, dim=-1)
            scores = (1 - penalty_alpha) * top_k_probs

            sampled_id: Optional[int] = None
            sampled_id_to_next_logits = {}
            sampled_id_to_next_hidden_states = {}

            visited_sampled_ids: Set[int] = set()

            if prev_norm_hidden_states_pos < prev_norm_hidden_states.size(0):
                # [T, hidden_size]
                prev_norm_hidden_states = prev_norm_hidden_states[:prev_norm_hidden_states_pos]

            for top_k_idx in range(top_k):
                idx_in_top_k = torch.argmax(scores)
                cur_sampled_id = int(top_k_indices[idx_in_top_k])

                if cur_sampled_id in visited_sampled_ids:
                    # Search previously, meaning this one has the highest score.
                    sampled_id = cur_sampled_id
                    break
                visited_sampled_ids.add(cur_sampled_id)

                # Get next logits.
                input_ids[0][0] = cur_sampled_id
                cur_next_logits, cur_next_hidden_states = \
                    self.model_inference.model_decode_one_token(
                        input_pos=input_pos,
                        input_ids=input_ids,
                    )

                # [1, hidden_size]
                cur_next_hidden_states = cur_next_hidden_states[0, -1].unsqueeze(0)
                cur_next_hidden_states /= cur_next_hidden_states.norm()

                # [1, T]
                cosine_matrix = torch.matmul(
                    cur_next_hidden_states,
                    prev_norm_hidden_states.transpose(0, 1),
                )
                similarity = torch.max(cosine_matrix)
                # Update degeneration penalty.
                scores[idx_in_top_k] -= penalty_alpha * similarity

                sampled_id_to_next_logits[cur_sampled_id] = cur_next_logits
                sampled_id_to_next_hidden_states[cur_sampled_id] = cur_next_hidden_states

                if similarity < similarity_thr:
                    # Most similarity should be small.
                    sampled_id = cur_sampled_id
                    break

                if top_k_idx == top_k - 1:
                    # Last one.
                    sampled_id = cur_sampled_id

            assert sampled_id is not None

            if sampled_id == self.eos_token:
                break
            sampled_ids.append(sampled_id)

            logits = sampled_id_to_next_logits[sampled_id]

            hidden_states = sampled_id_to_next_hidden_states[sampled_id]
            mod_pos = prev_norm_hidden_states_pos % prev_norm_hidden_states.size(0)
            prev_norm_hidden_states[mod_pos] = hidden_states[0]
            prev_norm_hidden_states_pos += 1

            input_pos += 1

        return Response(
            succeeded=True,
            error_message='',
            content=self.model_inference.tokenizer_decode(sampled_ids),
            num_prompt_tokens=num_prompt_tokens,
            num_completion_tokens=len(sampled_ids),
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

        logits, hidden_states, num_prompt_tokens = model_prefill_result

        if not generation_config.do_sample:
            if generation_config.contrastive_penalty_alpha > 0:
                func_strategy = self.strategy_contrastive
            else:
                func_strategy = self.strategy_greedy
        else:
            func_strategy = self.strategy_sample

        with torch.inference_mode():
            return func_strategy(
                logits=logits,
                hidden_states=hidden_states,
                num_prompt_tokens=num_prompt_tokens,
                generation_config=generation_config,
            )
