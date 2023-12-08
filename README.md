# gpt-blazing

POC for now.

GPU: 3090
Baichuan2 13b q8 (this project): 45.9 tok/s (on par with llama.cpp)
Baichuan2 13b q8 (huggingface): 7.9 tok/s
Llama2 13b q8: 55.01 tok/s (--compile); 55.50 tok/s (--compile & --compile_prefill)

Input:

```txt
input_ids = tokenizer.chat_tokenize([('user', "帮我写一篇与A股主题相关的作文，800字左右")])
```

Log of Baichuan2 13b q8.
Default config.

```txt
Loading...
Compiling...
/usr/lib/python3/dist-packages/requests/__init__.py:89: RequestsDependencyWarning: urllib3 (2.1.0) or chardet (3.0.4) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
[2023-12-08 08:37:06,280] torch._dynamo.convert_frame.__recompiles: [DEBUG] Current config does not match config saved when compiling
[2023-12-08 08:37:06,280] torch._dynamo.convert_frame.__recompiles: [DEBUG] Saved hash: d6d358f, Current hash: 50ec1ef
[2023-12-08 08:37:06,280] torch._dynamo.convert_frame.__recompiles: [DEBUG] Restoring saved config.
[2023-12-08 08:37:06,280] torch._dynamo.convert_frame.__recompiles: [DEBUG] * assume_static_by_default=False (prev: True)
prefill compiling time: 90.73871875 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
prefill compiling time: 0.08909593963623047 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
prefill compiling time: 0.08852252960205079 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
prefill compiling time: 0.08801420593261719 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
prefill compiling time: 0.08796966552734375 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 27.943076171875 tensor([0], device='cuda:0', dtype=torch.int32) tensor([[113774]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.45230364990234373 tensor([1], device='cuda:0', dtype=torch.int32) tensor([[3421]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.02133568000793457 tensor([2], device='cuda:0', dtype=torch.int32) tensor([[55520]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021333248138427734 tensor([3], device='cuda:0', dtype=torch.int32) tensor([[60405]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021325599670410155 tensor([4], device='cuda:0', dtype=torch.int32) tensor([[67167]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.02133478355407715 tensor([5], device='cuda:0', dtype=torch.int32) tensor([[11442]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.02133955192565918 tensor([6], device='cuda:0', dtype=torch.int32) tensor([[70273]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021355680465698242 tensor([7], device='cuda:0', dtype=torch.int32) tensor([[106250]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021395328521728516 tensor([8], device='cuda:0', dtype=torch.int32) tensor([[62685]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.02137763214111328 tensor([9], device='cuda:0', dtype=torch.int32) tensor([[49434]], device='cuda:0', dtype=torch.int32)
Running...
input_ids: [1, 195, 29724, 91529, 92489, 92343, 93176, 3824, 9820, 8419, 65, 92370, 92335, 92335, 92845, 3115, 196]
prefill_dt_delta: 0.095547
decode_dt_delta: 0:00:10.208315
tok/s: 46.33477709102824
encode_decode_dt_delta: 10.303892
tok/s: 45.90498425255234
Generation:
A股市场：机遇与挑战并存

随着中国经济的快速发展，A股市场已经成为全球投资者关注的焦点。作为全球最大的新兴市场，A股市场不仅为中国企业提供了融资的平台，也为全球投资者提供了丰富的投资机会。然而，A股市场的发展并非一帆风顺，面临着诸多机遇与挑战。本文将从以下几个方面探讨A股市场的现状与发展。

首先，A股市场的规模不断扩大。近年来，中国股市的市值不断攀升，已经成为全球第二大股市。据统计，截至2020年底，A股市场的总市值已经超过100万亿元人民币，占全球 股市总市值的比重超过16%。这一成绩的取得，得益于中国经济的高质量发展，以及资本市场改革的不断深化。随着中国经济的持续增长，A股市场的规模有望继续扩大，为全球投资者提供更多的投资机会。

其次，A股市场的投资者结构不断优化。过去，A股市场的投资者主要以散户为主，这使得市场容易出现过度投机和泡沫现象。然而，随着资本市场改革的深入推进，A股市场的 投资者结构正在发生变化。越来越多的机构投资者进入市场，成为市场的主导力量。据统计，截至2020年底，A股市场的机构投资者占比已经超过60%。这些机构投资者具有较强的风险控制能力和投资决策能力，有助于提高市场的稳定性和透明度。

然而，A股市场的发展也面临着诸多挑战。首先，市场波动性较大。近年来，A股市场波动性较大，市场风险不容忽视。这主要与市场参与者结构、市场制度设计等因素有关。为了降低市场波动性，监管部门需要进一步完善市场制度，加强市场监管，提高市场的透明度和稳定性。

其次，市场估值水平相对较高。与全球其他主要股市相比，A股市场的估值水平普遍较高。这一现象可能与市场投资者结构、市场制度等因素有关。为了降低市场估值水平，监 管部门需要进一步推进资本市场改革，优化市场环境，引导投资者理性投资。

最后，市场法治化水平有待提高。近年来，A股市场违法违规行为时有发生，市场法治化水平亟待提高。为了提高市场法治化水平，监管部门需要加大执法力度，严厉打击违法 违规行为，保护投资者合法权益。

总之，A股市场作为全球投资者关注的焦点，既面临着巨大的发展机遇，也面临着诸多挑战。只有通过深化改革，加强监管，优化市场环境，才能确保A股市场的健康发展，为全球投资者提供更多的投资机会。
```

Log of Baichuan2 13b q8.
use_original_attn_impl=True.

```txt
08:10:49 wden@node17 gpt-blazing ±|master ✗|→ fib gpt_blazing_model/baichuan2/debug.py:debug_greedy_decoding_performance
Loading...
Compiling...
/usr/lib/python3/dist-packages/requests/__init__.py:89: RequestsDependencyWarning: urllib3 (2.1.0) or chardet (3.0.4) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
[2023-12-08 08:41:34,354] torch._dynamo.convert_frame.__recompiles: [DEBUG] Current config does not match config saved when compiling
[2023-12-08 08:41:34,354] torch._dynamo.convert_frame.__recompiles: [DEBUG] Saved hash: 3e78a38, Current hash: ba699d0
[2023-12-08 08:41:34,354] torch._dynamo.convert_frame.__recompiles: [DEBUG] Restoring saved config.
[2023-12-08 08:41:34,354] torch._dynamo.convert_frame.__recompiles: [DEBUG] * assume_static_by_default=False (prev: True)
prefill compiling time: 37.127765625 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
prefill compiling time: 0.08430150604248046 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
prefill compiling time: 0.08394572448730468 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
prefill compiling time: 0.08386239624023438 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
prefill compiling time: 0.08378777313232422 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 30.09446484375 tensor([0], device='cuda:0', dtype=torch.int32) tensor([[115539]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.23042994689941407 tensor([1], device='cuda:0', dtype=torch.int32) tensor([[125191]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021494592666625977 tensor([2], device='cuda:0', dtype=torch.int32) tensor([[69103]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021598527908325196 tensor([3], device='cuda:0', dtype=torch.int32) tensor([[80459]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021652320861816406 tensor([4], device='cuda:0', dtype=torch.int32) tensor([[53423]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.02174038314819336 tensor([5], device='cuda:0', dtype=torch.int32) tensor([[60878]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021850879669189454 tensor([6], device='cuda:0', dtype=torch.int32) tensor([[116117]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021874528884887695 tensor([7], device='cuda:0', dtype=torch.int32) tensor([[1881]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021878368377685548 tensor([8], device='cuda:0', dtype=torch.int32) tensor([[108769]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021875616073608398 tensor([9], device='cuda:0', dtype=torch.int32) tensor([[10937]], device='cuda:0', dtype=torch.int32)
Running...
input_ids: [1, 195, 29724, 91529, 92489, 92343, 93176, 3824, 9820, 8419, 65, 92370, 92335, 92335, 92845, 3115, 196]
prefill_dt_delta: 0.091461
decode_dt_delta: 0:00:10.447549
tok/s: 45.27377665326097
encode_decode_dt_delta: 10.539041
tok/s: 44.880743893111344
Generation:
A股市场：机遇与挑战并存

随着中国经济的快速发展，A股市场已经成为全球投资者关注的焦点。作为全球最大的新兴市场，A股市场不仅为中国企业提供了融资的平台，也为全球投资者提供了丰富的投资机会。然而，A股市场的发展并非一帆风顺，面临着诸多机遇与挑战。本文将从以下几个方面探讨A股市场的现状与发展。

首先，A股市场的规模不断扩大。近年来，中国股市的市值不断攀升，已经成为全球第二大股市。据统计，截至2020年底，A股市场的总市值已经超过100万亿元人民币，占全球 股市总市值的比重超过16%。这一成绩的取得，得益于中国经济的高质量发展，以及资本市场改革的不断深化。随着中国经济的持续增长，A股市场的规模有望继续扩大，为全球投资者提供更多的投资机会。

其次，A股市场的投资者结构不断优化。过去，A股市场的投资者主要以散户为主，这使得市场容易出现过度投机和泡沫现象。然而，随着资本市场改革的深入推进，A股市场的 投资者结构正在发生变化。越来越多的机构投资者进入市场，成为市场的主导力量。据统计，截至2020年底，A股市场的机构投资者占比已经超过60%。这些机构投资者具有较强的风险控制能力和投资决策能力，有助于提高市场的稳定性和透明度。

然而，A股市场的发展也面临着诸多挑战。首先，市场波动性较大。近年来，A股市场波动性较大，市场风险不容忽视。这主要与市场参与者结构、市场制度设计等因素有关。为了降低市场波动性，监管部门需要进一步完善市场制度，加强市场监管，提高市场的透明度和稳定性。

其次，市场估值水平相对较高。与全球其他主要股市相比，A股市场的估值水平普遍较高。这一现象可能与市场投资者结构、市场制度等因素有关。为了降低市场估值水平，监 管部门需要进一步推进资本市场改革，优化市场环境，引导投资者理性投资。

最后，市场法治化水平有待提高。近年来，A股市场违法违规行为时有发生，市场法治化水平亟待提高。为了提高市场法治化水平，监管部门需要加大执法力度，严厉打击违法 违规行为，保护投资者合法权益。

总之，A股市场作为全球投资者关注的焦点，既面临着巨大的发展机遇，也面临着诸多挑战。只有通过深化改革，加强监管，优化市场环境，才能确保A股市场的健康发展，为全球投资者提供更多的投资机会。
```

Log of Baichuan2 13b q8.
huggingface.

```txt
07:55:01 wden@node17 gpt-blazing ±|master ✗|→ fib gpt_blazing_model/baichuan2/debug.py:eval_hf_model
/usr/lib/python3/dist-packages/requests/__init__.py:89: RequestsDependencyWarning: urllib3 (2.1.0) or chardet (3.0.4) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
/home/wden/.local/lib/python3.8/site-packages/transformers/utils/generic.py:311: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
Xformers is not installed correctly. If you want to use memory_efficient_attention to accelerate training use the following command to install Xformers
pip install xformers.
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:20<00:00,  6.89s/it]
Warmup
/home/wden/.local/lib/python3.8/site-packages/transformers/generation/configuration_utils.py:362: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.3` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
/home/wden/.local/lib/python3.8/site-packages/transformers/generation/configuration_utils.py:367: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.85` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(
/home/wden/.local/lib/python3.8/site-packages/transformers/generation/configuration_utils.py:377: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `5` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
你好今天我能为您提供什么帮助？
Running...
decode_dt_delta: 0:01:07.719957
tok/s: 7.900182216595324
题目：A股市场的机遇与挑战

随着中国经济的快速发展，A股市场已经成为全球资本市场的重要组成部分。然而，在这个充满机遇的市场中，也面临着诸多挑战。本文将探讨A股市场的机遇与挑战，以及如何应对这些挑战，实现市场的健康发展。

一、A股市场的机遇

1. 经济增长的驱动力

中国经济持续高速增长，为A股市场提供了强大的支撑。近年来，中国GDP增速稳居世界前列，成为全球经济的重要引擎。这为A股市场带来了巨大的投资机会，吸引了大量国内 外投资者。

2. 政策红利

政府对资本市场的重视程度不断提高，出台了一系列政策措施支持A股市场的发展。例如，注册制改革、科创板设立、金融供给侧结构性改革等，都为A股市场带来了新的发展机遇。

3. 科技创新驱动

科技创新是当今世界发展的关键驱动力，而中国正致力于建设创新型国家。A股市场已成为科技创新企业上市融资的主要平台，为投资者提供了丰富的投资选择。同时，科技创 新企业的快速发展也为A股市场带来了新的增长点。

二、A股市场的挑战

1. 市场波动性较大

A股市场波动性较大，投资者风险承受能力要求较高。在市场波动过程中，投资者容易出现恐慌情绪，导致市场出现非理性下跌。此外，市场波动性较大也容易引发系统性金融 风险。

2. 上市公司质量参差不齐

虽然A股市场涌现出一批优秀的上市公司，但整体质量仍有待提高。一些上市公司存在信息披露不透明、内部治理不规范等问题，影响了市场的形象和投资者信心。

3. 投资者结构不合理

目前，A股市场的投资者结构仍以散户为主，机构投资者占比相对较低。这种投资者结构容易导致市场过度投机，加大了市场的风险。

三、应对策略

1. 加强市场监管

监管部门应加强对市场的监管，打击违法违规行为，维护市场秩序。同时，推动市场改革，完善市场制度，为市场发展创造良好的环境。

2. 提高上市公司质量

上市公司应注重自身发展，提高盈利能力，确保信息披露的真实性、准确性和及时性。同时，加强内部治理，提升公司治理水平，树立良好的市场形象。

3. 优化投资者结构

鼓励更多长期投资者参与A股市场，引导投资者树立正确的投资观念，降低市场投机性。同时，大力发展机构投资者，提高其在市场中的比重，发挥其在市场稳定中的作用。

总之，A股市场在面临机遇的同时，也面临着诸多挑战。只有积极应对挑战，不断优化市场环境，才能使A股市场实现可持续发展，为中国经济注入更多的活力。
```
