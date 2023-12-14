from typing import Any
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import time
import asyncio

import iolite as io

from gpt_blazing.model.interface import Role
from gpt_blazing.engine import Engine
from gpt_blazing.engine_pool import EnginePool
from gpt_blazing.model.baichuan2.inference import (
    Baichuan2ModelInferenceConfig,
    Baichuan2ModelInference,
)


class Globals:
    engine: Any = None


def mock_initializer():
    current = multiprocessing.current_process()
    print(current, current._identity)

    worker_idx = current._identity[-1]

    print('worker_idx =', worker_idx)
    engine = Engine(
        Baichuan2ModelInference(
            Baichuan2ModelInferenceConfig(
                model_folder=str(
                    io.folder('$GPT_BLAZING_DATA/model/baichuan2-13b-chat/', expandvars=True)
                ),
                device=f'cuda:{worker_idx}',
            )
        )
    )
    print('done')
    Globals.engine = engine


def touch():
    print(Globals.engine)
    return os.getpid()


def debug_process_pool_executor_init():
    pool = ProcessPoolExecutor(
        max_workers=2,
        initializer=mock_initializer,
        mp_context=multiprocessing.get_context('spawn'),
    )
    f0 = pool.submit(touch)
    f1 = pool.submit(touch)
    print(f0, f1)
    breakpoint()


def mock_initializer_rlock(lock: Any, counter: Any):
    with lock:
        worker_idx = counter.value
        counter.value += 1
        print('worker_idx =', worker_idx)
        current = multiprocessing.current_process()
        print(current)
        print('Sleeping 5')
        time.sleep(5)
    print(current, 'done!')


def debug_rlock():
    manager = multiprocessing.Manager()
    pool = ProcessPoolExecutor(
        max_workers=2,
        initializer=mock_initializer_rlock,
        initargs=(
            manager.Lock(),
            manager.Value('i', 0),
        ),
        mp_context=multiprocessing.get_context('spawn'),
    )
    print(pool)
    f0 = pool.submit(pow, 10, 2)
    print(f0)
    breakpoint()


def mock_initializer_condition(condition: Any, counter: Any):
    current = multiprocessing.current_process()
    with condition:
        print(current)
        counter.value += 1
        condition.notify()
        time.sleep(1)
        print(current, 'done')


def debug_condition():
    manager = multiprocessing.Manager()
    condition = manager.Condition()
    counter = manager.Value('i', 0)
    pool = ProcessPoolExecutor(
        max_workers=10,
        initializer=mock_initializer_condition,
        initargs=(
            condition,
            counter,
        ),
        mp_context=multiprocessing.get_context('spawn'),
    )
    print(pool)
    pool.submit(pow, 10, 2)

    with condition:
        print('Waiting...')
        condition.wait_for(lambda: counter.value == 10)
        print('Done!')

    breakpoint()


async def debug_async_generate(pool: EnginePool):
    r0 = await pool.generate([(Role.USER, "帮我写一篇与A股主题相关的作文，800字左右")])
    print(r0.content)

    r1 = pool.generate([(Role.USER, "帮我写一篇与A股主题相关的作文，800字左右")])
    r2 = pool.generate([(Role.USER, "帮我写一篇与AI相关的作文，800字左右")])
    print(r1)
    print(r2)
    results = await asyncio.gather(r1, r2)
    print(results)
    breakpoint()


def debug_engine_pool():
    pool = EnginePool(
        model_inference=Baichuan2ModelInference(
            Baichuan2ModelInferenceConfig(
                model_folder=str(
                    io.folder('$GPT_BLAZING_DATA/model/baichuan2-13b-chat/', expandvars=True)
                )
            )
        ),
        devices=['cuda:0', 'cuda:1'],
    )
    print(pool)
    breakpoint()
    asyncio.run(debug_async_generate(pool))


if __name__ == '__main__':
    debug_process_pool_executor_init()
