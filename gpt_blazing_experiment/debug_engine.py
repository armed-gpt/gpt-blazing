from typing import List, Tuple
from datetime import datetime

import iolite as io

from gpt_blazing.engine import Engine
from gpt_blazing.model.interface import Role
from gpt_blazing.model.baichuan2.inference import (
    Baichuan2ModelInferenceConfig,
    Baichuan2ModelInference,
)


def create_engine():
    model_inference = Baichuan2ModelInference(
        Baichuan2ModelInferenceConfig(
            model_folder=str(
                io.folder('$GPT_BLAZING_DATA/model/baichuan2-13b-chat/', expandvars=True)
            ),
            device='cuda:0',
            use_dynamic_dispatch=True,
        )
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
    print('generate:', generate_total_seconds, response.completion_tokens / generate_total_seconds)

    print(response.content)


def run_demo():
    engine = create_engine()

    rounds: List[Tuple[Role, str]] = []
    while True:
        content = input('[USER]: ').strip()
        if content == 'reset':
            rounds = []
            continue
        rounds.append((Role.USER, content))
        generate_dt_begin = datetime.now()
        try:
            response = engine.generate(rounds)
        except Exception:
            print('engine.generate failed.')
            rounds = []
            continue
        generate_dt_end = datetime.now()
        generate_total_seconds = (generate_dt_end - generate_dt_begin).total_seconds()
        print(
            f'[ASSISTANT]: {response.content}\n'
            f'(secs={generate_total_seconds}, '
            f'prompt_tokens={response.prompt_tokens}, '
            f'completion_tokens={response.completion_tokens}, '
            f'tok/s={response.completion_tokens/generate_total_seconds:.1f})'
        )
        rounds.append((Role.ASSISTANT, response.content))
