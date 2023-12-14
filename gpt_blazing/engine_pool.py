from typing import Optional, Sequence, Tuple, Any, cast
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from datetime import datetime
import asyncio
import logging

from gpt_blazing.engine import Engine, GenerationConfig
from gpt_blazing.model.interface import ModelInference, Role

logger = logging.getLogger(__name__)


class Globals:
    engine: Any = None


def _initialize_engine(
    worker_lock: Any,
    worker_counter: Any,
    condition: Any,
    succeeded_counter: Any,
    finished_counter: Any,
    model_inference: ModelInference,
    devices: Sequence[str],
):
    logging.basicConfig(
        format='%(asctime)s - (%(name)s)[%(process)d][%(levelname)s]: %(message)s',
        level='INFO',
        force=True,
    )

    try:
        with worker_lock:
            # 1. Load model one by one to avoid I/O & OOM issue.
            worker_idx: int = worker_counter.value
            worker_counter.value += 1
            device = devices[worker_idx]

            logger.info(f'Loading model in worker_idx={worker_idx} with device={device}.')
            model_inference.load_model(device)
            logger.info(f'worker_idx={worker_idx} with device={device} model loaded!')

            # 2. Compile model.
            # NOTE: compiling model in parallel could trigger CUDA error.
            logger.info(f'Compiling model in worker_idx={worker_idx}.')
            model_inference.compile_model()
            assert model_inference.model_is_ready()
            logger.info(f'worker_idx={worker_idx} model compiled!')

            Globals.engine = Engine(model_inference)
            logger.info(f'worker_idx={worker_idx} engine initialized!')

            with condition:
                succeeded_counter.value += 1

    except Exception:
        logger.exception('Something is wrong.')

    finally:
        logger.info('Finished.')
        with condition:
            finished_counter.value += 1
            condition.notify()


def _touch_engine():
    assert Globals.engine


def _engine_generate(
    rounds: Sequence[Tuple[Role, str]],
    generation_config: Optional[GenerationConfig],
):
    return cast(Engine, Globals.engine).generate(
        rounds=rounds,
        generation_config=generation_config,
    )


class EnginePool:

    def __init__(
        self,
        model_inference: ModelInference,
        devices: Sequence[str],
    ):
        self.manager = multiprocessing.Manager()
        condition = self.manager.Condition()
        succeeded_counter = self.manager.Value('i', 0)
        finished_counter = self.manager.Value('i', 0)
        self.executor = ProcessPoolExecutor(
            max_workers=len(devices),
            initializer=_initialize_engine,
            initargs=(
                # NOTE: Manager.Value does not have a lock.
                # see https://github.com/python/cpython/issues/79967
                self.manager.Lock(),
                self.manager.Value('i', 0),
                condition,
                succeeded_counter,
                finished_counter,
                model_inference,
                devices,
            ),
            # NOTE: CUDA does not support fork mode.
            mp_context=multiprocessing.get_context('spawn'),
        )
        # NOTE: processes are created in a deferred manner, hence trigger the creation manually.
        self.executor.submit(_touch_engine)

        init_dt_begin = datetime.now()
        logger.info('Initializing EnginePool...')
        with condition:
            condition.wait_for(lambda: finished_counter.value == len(devices))
            assert succeeded_counter.value == len(devices)
        logger.info(
            'EnginePool initialized: '
            f'{(datetime.now() - init_dt_begin).total_seconds()}s'
        )

    async def generate(
        self,
        rounds: Sequence[Tuple[Role, str]],
        generation_config: Optional[GenerationConfig] = None,
    ):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            _engine_generate,
            rounds,
            generation_config,
        )
