from typing import TypeVar, Generic, Optional, Callable, Any, Sequence, Tuple
from enum import unique, Enum

import torch


@unique
class QuantizationMode(Enum):
    Q8 = 'q8'


@unique
class Role(Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'

    @classmethod
    def from_string(cls, text: str):
        return _TEXT_TO_ROLE[text]


_TEXT_TO_ROLE = {role.value: role for role in Role}

_T_CONFIG = TypeVar('_T_CONFIG')


class ModelInference(Generic[_T_CONFIG]):

    def __init__(
        self,
        config: _T_CONFIG,
        func_process_model: Optional[Callable[[Any], None]] = None,
    ):
        self.config = config
        self.func_process_model = func_process_model

    def get_eos_token(self) -> int:
        raise NotImplementedError()

    def model_prefill(
        self,
        rounds: Sequence[Tuple[Role, str]],
        cache_system: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError()

    def model_decode_one_token(
        self,
        input_pos: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def tokenizer_decode(self, tokens: Sequence[int]) -> str:
        raise NotImplementedError()
