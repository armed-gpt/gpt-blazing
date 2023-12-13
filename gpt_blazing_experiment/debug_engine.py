from datetime import datetime

import iolite as io

from gpt_blazing.engine import Engine
from gpt_blazing.model.interface import Role
from gpt_blazing.model.baichuan2.inference import (
    Baichuan2ModelInferenceConfig,
    Baichuan2ModelInference,
)


def debug_engine():
    init_dt_begin = datetime.now()
    engine = Engine(
        Baichuan2ModelInference(
            Baichuan2ModelInferenceConfig(
                model_folder=str(
                    io.folder('$GPT_BLAZING_DATA/model/baichuan2-13b-chat/', expandvars=True)
                ),
                device='cuda:0',
            )
        )
    )
    init_dt_end = datetime.now()
    print('init:', (init_dt_end - init_dt_begin).total_seconds())

    generate_dt_begin = datetime.now()
    response = engine.generate([(Role.USER, "帮我写一篇与A股主题相关的作文，800字左右")])
    generate_dt_end = datetime.now()
    print('generate:', (generate_dt_end - generate_dt_begin).total_seconds())

    print(response)

    breakpoint()
