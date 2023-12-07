# gpt-blazing

POC for now.

Baichuan2 13b q8: ~45 tok/s (on par with llama.cpp)

Input:

```txt
input_ids = tokenizer.chat_tokenize([('user', "帮我写一篇与A股主题相关的作文，800字左右")])
```

Log:

```txt
04:45:08 wden@node17 gpt-blazing ±|master ✗|→ fib gpt_blazing_model/baichuan2/debug.py:debug_greedy_decoding_performance
Loading...
Compiling...
/usr/lib/python3/dist-packages/requests/__init__.py:89: RequestsDependencyWarning: urllib3 (2.1.0) or chardet (3.0.4) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
[2023-12-07 16:46:33,122] torch._dynamo.convert_frame.__recompiles: [DEBUG] Current config does not match config saved when compiling
[2023-12-07 16:46:33,122] torch._dynamo.convert_frame.__recompiles: [DEBUG] Saved hash: 6013f71, Current hash: e27661e
[2023-12-07 16:46:33,122] torch._dynamo.convert_frame.__recompiles: [DEBUG] Restoring saved config.
[2023-12-07 16:46:33,122] torch._dynamo.convert_frame.__recompiles: [DEBUG] * assume_static_by_default=False (prev: True)
prefill compiling time: 37.14701171875 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
prefill compiling time: 0.08483190155029297 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
prefill compiling time: 0.08404969787597656 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
prefill compiling time: 0.08385062408447265 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
prefill compiling time: 0.08378195190429688 tensor([0, 1, 2, 3], device='cuda:0', dtype=torch.int32) tensor([[ 195, 4330, 2908,   66]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 28.431486328125 tensor([0], device='cuda:0', dtype=torch.int32) tensor([[108955]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.22653610229492188 tensor([1], device='cuda:0', dtype=torch.int32) tensor([[100408]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021381824493408204 tensor([2], device='cuda:0', dtype=torch.int32) tensor([[125267]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.02140284729003906 tensor([3], device='cuda:0', dtype=torch.int32) tensor([[54827]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.02139244842529297 tensor([4], device='cuda:0', dtype=torch.int32) tensor([[107534]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.02143187141418457 tensor([5], device='cuda:0', dtype=torch.int32) tensor([[96381]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021445503234863283 tensor([6], device='cuda:0', dtype=torch.int32) tensor([[31898]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021447904586791994 tensor([7], device='cuda:0', dtype=torch.int32) tensor([[23418]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021486560821533204 tensor([8], device='cuda:0', dtype=torch.int32) tensor([[56880]], device='cuda:0', dtype=torch.int32)
decode_one_token compiling time: 0.021502239227294922 tensor([9], device='cuda:0', dtype=torch.int32) tensor([[113711]], device='cuda:0', dtype=torch.int32)
Running...
input_ids: [1, 195, 29724, 91529, 92489, 92343, 93176, 3824, 9820, 8419, 65, 92370, 92335, 92335, 92845, 3115, 196]
prefill_dt_delta: 0.091285
decode_dt_delta: 0:00:10.252947
tok/s: 46.13307764099434
encode_decode_dt_delta: 10.344261
tok/s: 45.725837737466215
Generation:
A股市场：机遇与挑战并存

随着中国经济的快速发展，A股市场已经成为全球投资者关注的焦点。作为全球最大的新兴市场，A股市场不仅为中国企业提供了融资的平台，也为全球投资者提供了丰富的投资机会。然 而，A股市场的发展并非一帆风顺，面临着诸多机遇与挑战。本文将从以下几个方面探讨A股市场的现状与发展。

首先，A股市场的规模不断扩大。近年来，中国股市的市值不断攀升，已经成为全球第二大股市。据统计，截至2020年底，A股市场的总市值已经超过100万亿元人民币，占全球股市总市值的比重超过16%。这一成绩的取得，得益于中国经济的高质量发展，以及资本市场改革的不断深化。随着中国经济的持续增长，A股市场的规模有望继续扩大，为全球投资者提供更多的投 资机会。

其次，A股市场的投资者结构不断优化。过去，A股市场的投资者主要以散户为主，这使得市场容易出现过度投机和泡沫现象。然而，随着资本市场改革的深入推进，A股市场的投资者结构正在发生变化。越来越多的机构投资者进入市场，成为市场的主导力量。据统计，截至2020年底，A股市场的机构投资者占比已经超过60%。这些机构投资者具有较强的风险控制能力和投 资决策能力，有助于提高市场的稳定性和透明度。

然而，A股市场的发展也面临着诸多挑战。首先，市场波动性较大。近年来，A股市场波动性较大，市场风险不容忽视。这主要与市场参与者结构、市场制度设计等因素有关。为了降低市 场波动性，监管部门需要进一步完善市场制度，加强市场监管，提高市场的透明度和稳定性。

其次，市场估值水平相对较高。与全球其他主要股市相比，A股市场的估值水平普遍较高。这一现象可能与市场投资者结构、市场制度等因素有关。为了降低市场估值水平，监管部门需要进一步推进资本市场改革，优化市场环境，引导投资者理性投资。

最后，市场法治化水平有待提高。近年来，A股市场违法违规行为时有发生，市场法治化水平亟待提高。为了提高市场法治化水平，监管部门需要加大执法力度，严厉打击违法违规行为，保护投资者合法权益。

总之，A股市场作为全球投资者关注的焦点，既面临着巨大的发展机遇，也面临着诸多挑战。只有通过深化改革，加强监管，优化市场环境，才能确保A股市场的健康发展，为全球投资者 提供更多的投资机会。
```
