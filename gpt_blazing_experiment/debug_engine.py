from typing import List, Tuple
from datetime import datetime
import logging

import iolite as io

from gpt_blazing.engine import Engine, GenerationConfig
from gpt_blazing.model.interface import Role
from gpt_blazing.model.baichuan2.inference import (
    Baichuan2ModelInferenceConfig,
    Baichuan2ModelInference,
)

logger = logging.getLogger(__name__)


def create_engine(**kwargs):  # type: ignore
    model_inference = Baichuan2ModelInference(
        Baichuan2ModelInferenceConfig(
            model_folder=str(
                io.folder('$GPT_BLAZING_DATA/model/baichuan2-13b-chat/', expandvars=True)
            ),
            device='cuda:0',
            **kwargs,
        ),
    )
    model_inference.load_model()
    model_inference.compile_model()
    return Engine(model_inference)


def debug_engine():
    init_dt_begin = datetime.now()
    engine = create_engine()
    init_dt_end = datetime.now()
    print('init:', (init_dt_end - init_dt_begin).total_seconds())

    generate_dt_begin = datetime.now()
    response = engine.generate([(Role.USER, "帮我写一篇与A股主题相关的作文，800字左右")])
    generate_dt_end = datetime.now()
    generate_total_seconds = (generate_dt_end - generate_dt_begin).total_seconds()
    print(
        'generate:',
        generate_total_seconds,
        response.num_completion_tokens / generate_total_seconds,
    )

    print(response.content)


def run_demo():
    # engine = create_engine(skip_torch_compile=True)
    engine = create_engine()

    generation_config = GenerationConfig(
        do_sample=False,
        contrastive_penalty_alpha=0.0,
        # top_k=6,
        # contrastive_penalty_alpha=0.6,
        # contrastive_similarity_thr=0.4,
    )
    # generation_config = GenerationConfig(do_sample=True)

    rounds: List[Tuple[Role, str]] = []
    while True:
        content = input('[USER]: ').strip()
        if content == 'reset':
            rounds = []
            continue
        rounds.append((Role.USER, content))
        generate_dt_begin = datetime.now()
        try:
            response = engine.generate(rounds, generation_config=generation_config)
        except Exception:
            logger.exception('engine.generate failed.')
            rounds = []
            continue
        generate_dt_end = datetime.now()
        generate_total_seconds = (generate_dt_end - generate_dt_begin).total_seconds()
        print(
            f'[ASSISTANT]: {response.content}\n'
            f'(secs={generate_total_seconds}, '
            f'num_prompt_tokens={response.num_prompt_tokens}, '
            f'num_completion_tokens={response.num_completion_tokens}, '
            f'tok/s={response.num_completion_tokens / generate_total_seconds:.1f})'
        )
        rounds.append((Role.ASSISTANT, response.content))
