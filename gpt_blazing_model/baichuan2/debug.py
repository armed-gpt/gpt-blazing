from typing import Tuple, Sequence, Optional
from datetime import datetime

import torch
import torch.nn.functional as F
import sentencepiece as spm
import iolite as io

from .model import (
    Baichuan2Model,
    Baichuan2ModelConfig,
    quantize_int8,
    EmptyInitOnDevice,
    load_model,
    model_prefill_2048,
    model_prefill_4096,
    compile_model_prefill,
    model_decode_one_token_2048,
    model_decode_one_token_4096,
    compile_model_decode_one_token,
    model_dispatch,
)
from .tokenizer import Baichuan2Tokenizer

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


def eval_hf_model():
    from transformers import AutoTokenizer
    from transformers.generation.utils import GenerationConfig

    with EmptyInitOnDevice():
        model = load_hf_model()
    model.generation_config = GenerationConfig.from_pretrained(BAICHUAN2_13B_MODEL_FOLDER)
    tokenizer = AutoTokenizer.from_pretrained(
        BAICHUAN2_13B_MODEL_FOLDER,
        use_fast=False,
        trust_remote_code=True,
    )

    model.generation_config.do_sample = False
    # pip install bitsandbytes scipy
    model = model.quantize(8).to('cuda:0')

    print('Warmup')
    with torch.inference_mode():
        print(model.chat(tokenizer, [{"role": "user", "content": '你好'}]))

    print('Running...')
    decode_dt_begin = datetime.now()
    with torch.inference_mode():
        response = model.chat(
            tokenizer,
            [{
                "role": "user",
                "content": "帮我写一篇与A股主题相关的作文，800字左右"
            }],
        )
    decode_dt_end = datetime.now()

    decode_dt_delta = decode_dt_end - decode_dt_begin
    print('decode_dt_delta:', decode_dt_delta)
    output_ids = tokenizer.encode(response, add_special_tokens=False)
    print('tok/s:', (len(output_ids) + 1) / decode_dt_delta.total_seconds())
    print(response)


def load_and_convert_to_model(model_folder: str = BAICHUAN2_13B_MODEL_FOLDER):
    with EmptyInitOnDevice():
        hf_model = load_hf_model(model_folder)
        model = Baichuan2Model(Baichuan2ModelConfig(debug=True))
        model.half()

        # NOTE: this is not working.
        # if not model.config.use_original_attn_impl:
        #     # For scaled_dot_product_attention.
        #     torch.backends.cuda.enable_flash_sdp(False)
        #     torch.backends.cuda.enable_mem_efficient_sdp(False)
        #     torch.backends.cuda.enable_math_sdp(True)

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
    input_ids = torch.tensor([[195, *input_ids]], dtype=torch.int)
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

    if not q8:
        move_model_to_devices(model, [('cuda:0', 0), ('cuda:1', 20)])  # type: ignore
    else:
        model = model.to('cuda:0')  # type: ignore

    if compile:
        print('Compiling...')
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    input_ids = generate_debug_input_ids(model_folder)
    input_ids = input_ids.to('cuda:0')
    input_pos = torch.arange(0, input_ids.shape[1], device=input_ids.device)
    with torch.inference_mode():
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        ):
            logits = model(input_pos=input_pos, end=2048, input_ids=input_ids)

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
# 0.9955
fib gpt_blazing_model/baichuan2/debug.py:compare_logits \
    --file0="$GPT_BLAZING_DATA/model/baichuan2/logits.pt" \
    --file1="$GPT_BLAZING_DATA/model/baichuan2/hf_logits.pt"

# 0.9943
fib gpt_blazing_model/baichuan2/debug.py:compare_logits \
    --file0="$GPT_BLAZING_DATA/model/baichuan2/logits.pt" \
    --file1="$GPT_BLAZING_DATA/model/baichuan2/q8_hf_logits.pt"

# 0.9955
fib gpt_blazing_model/baichuan2/debug.py:compare_logits \
    --file0="$GPT_BLAZING_DATA/model/baichuan2/logits.pt" \
    --file1="$GPT_BLAZING_DATA/model/baichuan2/compiled_logits.pt"

# 0.9945
fib gpt_blazing_model/baichuan2/debug.py:compare_logits \
    --file0="$GPT_BLAZING_DATA/model/baichuan2/logits.pt" \
    --file1="$GPT_BLAZING_DATA/model/baichuan2/compiled_q8_logits.pt"

# 0.9943
fib gpt_blazing_model/baichuan2/debug.py:compare_logits \
    --file0="$GPT_BLAZING_DATA/model/baichuan2/compiled_q8_logits.pt" \
    --file1="$GPT_BLAZING_DATA/model/baichuan2/q8_hf_logits.pt"
    '''
    logits0 = torch.load(file0, map_location='cuda:0')
    logits1 = torch.load(file1, map_location='cuda:0')

    tpsi0 = get_top_p_sorted_indices(logits0)
    tpsi1 = get_top_p_sorted_indices(logits1)

    rank = tpsi0 == tpsi1
    r = rank.sum() / rank.numel()
    print(r)


def demo_func(x: torch.Tensor, y: torch.Tensor, begin: int, end: int):
    return x[:, begin:end] + y[:, begin:end]


def debug_compile():
    func = torch.compile(demo_func, mode="reduce-overhead", fullgraph=True)

    x = torch.rand((1, 20, 128))
    y = torch.rand((1, 20, 128))

    print(func(x, y, 0, 10))
    # triggers Recompiling!
    print(func(x, y, 0, 15))
    print(func(x, y, 1, 2))


def timed(fn):  # type: ignore
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()  # type: ignore
    result = fn()
    end.record()  # type: ignore
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


def debug_greedy_decoding_performance():
    print('Loading...')
    model = load_model(
        model_pt=str(io.file("$GPT_BLAZING_DATA/model/baichuan2/13b-q8.pt", expandvars=True)),
        q8=True,
        config=Baichuan2ModelConfig(debug=False),
    )
    model.to('cuda:0')

    print('Compiling...')
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True

    input_ids = generate_debug_input_ids()
    input_ids = input_ids.to('cuda:0')

    prefill_2048 = compile_model_prefill(model_prefill_2048)
    prefill_4096 = compile_model_prefill(model_prefill_4096)

    for offset in [0, 2048]:
        for _ in range(3):
            input_pos = torch.arange(
                offset,
                offset + int(input_ids.shape[1]),
                device=input_ids.device,
                dtype=torch.int,
            )
            print(
                'prefill compiling time:',
                timed(
                    lambda: model_dispatch(
                        model=model,
                        func_2048=prefill_2048,
                        func_4096=prefill_4096,
                        input_pos=input_pos,
                        input_ids=input_ids,
                    )
                )[1],
            )

    import random

    decode_one_token_2048 = compile_model_decode_one_token(model_decode_one_token_2048)
    decode_one_token_4096 = compile_model_decode_one_token(model_decode_one_token_4096)
    input_pos = torch.tensor([0], device=input_ids.device, dtype=torch.int)

    for offset in [0, 2048]:
        for idx in range(3):
            input_pos[0] = offset + idx
            cur_input_ids = torch.tensor(
                [[random.randint(0, 125696)]],
                dtype=torch.int,
                device='cuda:0',
            )
            print(
                'decode_one_token compiling time:',
                timed(
                    lambda: model_dispatch(
                        model=model,
                        func_2048=decode_one_token_2048,
                        func_4096=decode_one_token_4096,
                        input_pos=input_pos,
                        input_ids=cur_input_ids,
                    )
                )[1],
            )

    print('Running...')
    tokenizer = Baichuan2Tokenizer(f'{BAICHUAN2_13B_MODEL_FOLDER}/tokenizer.model')
    _input_ids = tokenizer.chat_tokenize([('user', "帮我写一篇与A股主题相关的作文，800字左右")])
    print('input_ids:', _input_ids)
    input_ids = torch.tensor([_input_ids], dtype=torch.int)
    input_ids = input_ids.to('cuda:0')
    input_pos = torch.arange(0, int(input_ids.shape[1]), device=input_ids.device, dtype=torch.int)

    output_ids = []

    prefill_dt_begin = datetime.now()
    logits = model_dispatch(
        model=model,
        func_2048=prefill_2048,
        func_4096=prefill_4096,
        input_pos=input_pos,
        input_ids=input_ids,
    ).detach()
    logits = logits[:, -1]
    output_id = int(torch.argmax(logits, dim=1)[0])
    output_ids.append(output_id)
    prefill_dt_end = datetime.now()

    prefill_dt_delta = prefill_dt_end - prefill_dt_begin
    print('prefill_dt_delta:', prefill_dt_delta.total_seconds())

    decode_dt_begin = datetime.now()
    cur_input_ids = torch.tensor([[output_ids[0]]], dtype=torch.int, device='cuda:0')
    input_pos = torch.tensor([int(input_ids.shape[1])], device=input_ids.device, dtype=torch.int)

    # while input_pos[0] < 1024:
    while input_pos[0] < 1024:
        logits = model_dispatch(
            model=model,
            func_2048=decode_one_token_2048,
            func_4096=decode_one_token_4096,
            input_pos=input_pos,
            input_ids=cur_input_ids,
        ).detach()
        logits = logits[:, -1]
        output_id = torch.argmax(logits, dim=1)[0]
        output_ids.append(int(output_id))
        if output_ids[-1] == tokenizer.eos_token_id:
            break
        cur_input_ids[0] = output_id
        input_pos += 1
    decode_dt_end = datetime.now()

    decode_dt_delta = decode_dt_end - decode_dt_begin
    print('decode_dt_delta:', decode_dt_delta)
    print('tok/s:', len(output_ids) / decode_dt_delta.total_seconds())
    encode_decode_dt_delta = decode_dt_end - prefill_dt_begin
    print('encode_decode_dt_delta:', encode_decode_dt_delta.total_seconds())
    print('tok/s:', len(output_ids) / encode_decode_dt_delta.total_seconds())

    texts = []
    for token in output_ids[:-1]:
        texts.append(tokenizer.decode(token))
    print('Generation:')
    print(''.join(texts))


def export_model(
    output_file: str,
    model_folder: str = BAICHUAN2_13B_MODEL_FOLDER,
    q8: bool = False,
):
    '''
fib gpt_blazing_model/baichuan2/debug.py:export_model \
    --output_file="$GPT_BLAZING_DATA/model/baichuan2/13b-q8.pt" \
    --q8
    '''
    model = load_and_convert_to_model(model_folder)
    if q8:
        model = quantize_int8(model)

    torch.save(model.state_dict(), output_file)
