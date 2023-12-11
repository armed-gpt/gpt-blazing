from typing import TypeVar, Generic, Optional, Callable, Any
from enum import unique, Enum


@unique
class QuantizationMode(Enum):
    Q8 = 'q8'


_T_CONFIG = TypeVar('_T_CONFIG')


class Inference(Generic[_T_CONFIG]):

    def __init__(
        self,
        config: _T_CONFIG,
        func_process_model: Optional[Callable[[Any], None]] = None,
    ):
        self.config = config
        self.func_process_model = func_process_model
