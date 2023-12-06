from typing import Tuple, Sequence, Optional

import torch
import torch.nn.functional as F
import torch.utils._device
import sentencepiece as spm
import iolite as io

from .model import Baichuan2Model, Baichuan2ModelConfig, quantize_int8


# http://www.lernapparat.de/faster-model-init
class EmptyInitOnDevice(torch.overrides.TorchFunctionMode):  # type: ignore

    def __init__(self, device=None):  # type: ignore
        self.device = device

    def __torch_function__(self, func, types, args=(), kwargs=None):  # type: ignore
        kwargs = kwargs or {}
        if getattr(func, '__module__', None) == 'torch.nn.init':
            if 'tensor' in kwargs:
                return kwargs['tensor']
            else:
                return args[0]
        device_constructors = torch.utils._device._device_constructors()  # type: ignore
        if (
            self.device is not None and func in device_constructors and kwargs.get('device') is None
        ):
            kwargs['device'] = self.device
        return func(*args, **kwargs)


BAICHUAN2_13B_MODEL_FOLDER = str(
    io.folder(
        '$GPT_BLAZING_DATA/base/Baichuan2-13B-Chat',
        expandvars=True,
    )
)


def load_hf_model(
    model_folder: str = BAICHUAN2_13B_MODEL_FOLDER,
    device_map: Optional[str] = None,
):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_folder,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device_map,
    )


def load_and_convert_to_model(model_folder: str = BAICHUAN2_13B_MODEL_FOLDER):
    with EmptyInitOnDevice():
        hf_model = load_hf_model(model_folder)
        model = Baichuan2Model(Baichuan2ModelConfig(debug=True))
        model.half()

        if not model.config.use_original_attn_impl:
            # For scaled_dot_product_attention.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

    baichuan_model = hf_model.model

    model.embed_tokens.load_state_dict(baichuan_model.embed_tokens.state_dict())
    for layer_idx, layer in enumerate(model.layers):
        layer.load_state_dict(baichuan_model.layers[layer_idx].state_dict())
    model.norm.load_state_dict(baichuan_model.norm.state_dict())
    model.lm_head.load_state_dict(hf_model.lm_head.state_dict())
    return model


def generate_debug_input_ids(model_folder: str = BAICHUAN2_13B_MODEL_FOLDER):
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(f'{model_folder}/tokenizer.model')

    input_ids = sp_model.tokenize('测试一下。')  # type: ignore
    input_ids = torch.LongTensor([[195, *input_ids]])
    return input_ids


def move_model_to_devices(
    model: Baichuan2Model,
    device_and_layer_begin_pairs: Sequence[Tuple[str, int]],
):
    assert device_and_layer_begin_pairs
    device0 = device_and_layer_begin_pairs[0][0]
    model.embed_tokens.to(device0)
    model.alibi_mask = model.alibi_mask.to(device0)

    for pair_idx, (device, layer_begin) in enumerate(device_and_layer_begin_pairs):
        if pair_idx + 1 < len(device_and_layer_begin_pairs):
            layer_end = device_and_layer_begin_pairs[pair_idx + 1][1]
        else:
            layer_end = len(model.layers)

        for layer_idx in range(layer_begin, layer_end):
            model.layers[layer_idx].to(device)

    device1 = device_and_layer_begin_pairs[-1][0]
    model.norm.to(device1)
    model.lm_head.to(device1)


def save_model_logits(
    output_file: str,
    model_folder: str = BAICHUAN2_13B_MODEL_FOLDER,
    compile: bool = False,
    q8: bool = False,
):
    '''
fib gpt_blazing_model/baichuan2/debug.py:save_model_logits \
    --output_file="$GPT_BLAZING_DATA/model/baichuan2/logits.pt"

fib gpt_blazing_model/baichuan2/debug.py:save_model_logits \
    --output_file="$GPT_BLAZING_DATA/model/baichuan2/q8_logits.pt" \
    --q8

fib gpt_blazing_model/baichuan2/debug.py:save_model_logits \
    --output_file="$GPT_BLAZING_DATA/model/baichuan2/compiled_logits.pt" \
    --compile

fib gpt_blazing_model/baichuan2/debug.py:save_model_logits \
    --output_file="$GPT_BLAZING_DATA/model/baichuan2/compiled_q8_logits.pt" \
    --compile \
    --q8
    '''
    print('Loading...')
    model = load_and_convert_to_model(model_folder)

    if q8:
        print('Quantizing...')
        model = quantize_int8(model)

    if compile:
        print('Compiling...')
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    if not q8:
        move_model_to_devices(model, [('cuda:0', 0), ('cuda:1', 20)])  # type: ignore
    else:
        model = model.to('cuda:0')  # type: ignore

    input_ids = generate_debug_input_ids(model_folder)
    input_ids = input_ids.to('cuda:0')
    input_pos = torch.arange(0, input_ids.size(1), device=input_ids.device)
    with torch.inference_mode():
        logits = model(input_pos=input_pos, input_ids=input_ids)

    print('Saving to', output_file)
    torch.save(logits, output_file)


def save_hf_model_logits(
    output_file: str,
    model_folder: str = BAICHUAN2_13B_MODEL_FOLDER,
    q8: bool = False,
):
    '''
fib gpt_blazing_model/baichuan2/debug.py:save_hf_model_logits \
    --output_file="$GPT_BLAZING_DATA/model/baichuan2/hf_logits.pt"

fib gpt_blazing_model/baichuan2/debug.py:save_hf_model_logits \
    --output_file="$GPT_BLAZING_DATA/model/baichuan2/q8_hf_logits.pt" \
    --q8
    '''
    print('Loading...')
    with EmptyInitOnDevice():
        if not q8:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
            hf_model = load_hf_model(model_folder, device_map='auto')
        else:
            hf_model = load_hf_model(model_folder)
            hf_model = hf_model.quantize(8).to('cuda:0')  # type: ignore

    input_ids = generate_debug_input_ids(model_folder)
    input_ids = input_ids.to('cuda:0')
    with torch.inference_mode():
        output = hf_model.forward(input_ids=input_ids)

    print('Saving to', output_file)
    torch.save(output.logits, output_file)


def get_top_p_sorted_indices(logits: torch.Tensor, top_p: float = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumulative_probs <= top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = True
    sorted_indices[~mask] = -1
    return sorted_indices


def compare_logits(file0: str, file1: str):
    '''
# 1.
fib gpt_blazing_model/baichuan2/debug.py:compare_logits \
    --file0="$GPT_BLAZING_DATA/model/baichuan2/logits.pt" \
    --file1="$GPT_BLAZING_DATA/model/baichuan2/hf_logits.pt"

# 0.9943
fib gpt_blazing_model/baichuan2/debug.py:compare_logits \
    --file0="$GPT_BLAZING_DATA/model/baichuan2/logits.pt" \
    --file1="$GPT_BLAZING_DATA/model/baichuan2/q8_hf_logits.pt"

# 0.9949
fib gpt_blazing_model/baichuan2/debug.py:compare_logits \
    --file0="$GPT_BLAZING_DATA/model/baichuan2/logits.pt" \
    --file1="$GPT_BLAZING_DATA/model/baichuan2/compiled_logits.pt"

# 0.9945
fib gpt_blazing_model/baichuan2/debug.py:compare_logits \
    --file0="$GPT_BLAZING_DATA/model/baichuan2/logits.pt" \
    --file1="$GPT_BLAZING_DATA/model/baichuan2/compiled_q8_logits.pt"
    '''
    logits0 = torch.load(file0, map_location='cuda:0')
    logits1 = torch.load(file1, map_location='cuda:0')

    tpsi0 = get_top_p_sorted_indices(logits0)
    tpsi1 = get_top_p_sorted_indices(logits1)

    rank = tpsi0 == tpsi1
    r = rank.sum() / rank.numel()
    print(r)
