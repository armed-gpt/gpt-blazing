# gpt-blazing

## Installation

```bash
pip install --pre torch==2.2.0.dev20231207 --index-url https://download.pytorch.org/whl/nightly/cu118
pip install --pre gpt-blazing
```

## Usage

### Download a supported model.

Supported models:

- 🤗 [baichuan2-13b-chat](https://huggingface.co/gpt-blazing/baichuan2-13b-chat)
- more to be supported...

### Run the following demo.

```python
from datetime import datetime

from gpt_blazing.engine import Engine
from gpt_blazing.model.interface import Role
from gpt_blazing.model.baichuan2.inference import (
    Baichuan2ModelInferenceConfig,
    Baichuan2ModelInference,
)


init_dt_begin = datetime.now()
engine = Engine(
    Baichuan2ModelInference(
        Baichuan2ModelInferenceConfig(
            model_folder='the path of model folder you just downloaded.',
            device='cuda:0',
        )
    )
)
init_dt_end = datetime.now()
print('init:', (init_dt_end - init_dt_begin).total_seconds())

generate_dt_begin = datetime.now()
response = engine.generate([(Role.USER, "帮我写一篇与A股主题相关的作文，800字左右")])
generate_dt_end = datetime.now()
generate_total_seconds = (generate_dt_end - generate_dt_begin).total_seconds()
print('generate:', generate_total_seconds, response.num_tokens / generate_total_seconds)

print(response.content)
```

## Performance

GPU: 3090

|       Model       |       Technique       | Tokens/Second |
|:-----------------:|:---------------------:|:-------------:|
| **Baichuan2 13b** | Q8 **(this project)** | 50.1          |
| Baichuan2 13b     | Q8 (huggingface)      | 7.9           |
| Llama2 13b        | Q8 (gpt-fast)         | 55.5          |
