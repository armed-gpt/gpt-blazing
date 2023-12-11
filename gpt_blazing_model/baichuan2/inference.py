from typing import Optional, Callable, Any
import logging
import random

import torch
import attrs
import iolite as io

from gpt_blazing_model.interface import Inference, QuantizationMode
from .model import (
    load_model,
    model_prefill_2048,
    model_prefill_4096,
    compile_model_prefill,
    model_decode_one_token_2048,
    model_decode_one_token_4096,
    compile_model_decode_one_token,
    model_dispatch,
    Baichuan2ModelConfig,
)
from .tokenizer import Baichuan2Tokenizer

logger = logging.getLogger(__name__)


def timed(fn):  # type: ignore
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()  # type: ignore
    result = fn()
    end.record()  # type: ignore
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


@attrs.define
class Baichuan2InferenceConfig:
    model_folder: str
    model_config: Baichuan2ModelConfig = attrs.field(factory=Baichuan2ModelConfig)
    quantization_mode: QuantizationMode = QuantizationMode.Q8
    device: Optional[str] = None


class Baichuan2Inference(Inference[Baichuan2InferenceConfig]):

    def __init__(
        self,
        config: Baichuan2InferenceConfig,
        func_process_model: Optional[Callable[[Any], None]] = None,
    ):
        super().__init__(config, func_process_model)
        logger.info(f'Initializing Baichuan2Inference(config={config})')

        model_fd = io.folder(config.model_folder, exists=True)

        # TODO: support more modes.
        assert config.quantization_mode == QuantizationMode.Q8

        model_pt = str(model_fd / f'{config.quantization_mode.value}.pt')
        logger.info(f'Loading model_pt={model_pt}')
        self.model = load_model(model_pt=model_pt, config=config.model_config, q8=True)
        logger.info('Model loaded.')

        tokenizer_model = str(model_fd / 'tokenizer.model')
        logger.info(f'Loading tokenizer_model={tokenizer_model}')
        self.tokenizer = Baichuan2Tokenizer(tokenizer_model)
        logger.info('Tokenizer loaded.')

        logger.info(f'Moving model to device={config.device}')
        self.model = self.model.to(config.device)

        if func_process_model is not None:
            logger.info('func_process_model is set, calling func_process_model(self.model)...')
            func_process_model(self.model)

        logger.info('Compiling model...')
        self.prefill_2048 = compile_model_prefill(model_prefill_2048)
        self.prefill_4096 = compile_model_prefill(model_prefill_4096)
        self.decode_one_token_2048 = compile_model_decode_one_token(model_decode_one_token_2048)
        self.decode_one_token_4096 = compile_model_decode_one_token(model_decode_one_token_4096)
        self.trigger_model_compilation()

    def trigger_model_compilation(self):
        import torch._dynamo.config
        import torch._inductor.config

        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True

        logger.info('Trigger prefill compilation.')
        input_ids = torch.tensor([self.tokenizer.tokenize('随便写点什么')], dtype=torch.int)
        input_ids = input_ids.to(self.config.device)

        for offset in [0, 2048]:
            logger.info(f'offset={offset}')
            for idx in range(5):
                input_pos = torch.arange(
                    offset,
                    offset + int(input_ids.shape[1]),
                    device=input_ids.device,
                    dtype=torch.int,
                )
                _, num_seconds = timed(
                    lambda: model_dispatch(
                        model=self.model,
                        func_2048=self.prefill_2048,
                        func_4096=self.prefill_4096,
                        input_pos=input_pos,
                        input_ids=input_ids,
                    )
                )
                logger.info(f'[{idx}]: prefill compilation: {num_seconds}s.')

        logger.info('Trigger decode_one_token compilation.')
        for offset in [0, 2048]:
            logger.info(f'offset={offset}')
            for idx in range(5):
                input_pos = torch.tensor([offset + idx], device=self.config.device, dtype=torch.int)
                input_ids = torch.tensor(
                    [[random.randint(0, self.config.model_config.vocab_size)]],
                    dtype=torch.int,
                    device=self.config.device,
                )

                _, num_seconds = timed(
                    lambda: model_dispatch(
                        model=self.model,
                        func_2048=self.decode_one_token_2048,
                        func_4096=self.decode_one_token_4096,
                        input_pos=input_pos,
                        input_ids=input_ids,
                    )
                )
                logger.info(f'[{idx}]: decode_one_token compilation: {num_seconds}s.')
