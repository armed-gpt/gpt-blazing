# gpt-blazing

POC for now.

GPU: 3090

|       Model       |       Technique       | Tokens/Second |
|:-----------------:|:---------------------:|:-------------:|
| **Baichuan2 13b** | Q8 **(this project)** | 50.4          |
| Baichuan2 13b     | Q8 (huggingface)      | 7.9           |
| Llama2 13b        | Q8 (gpt-fast)         | 55.5          |
