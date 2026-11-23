| changes vs baseline | params | seeds | val loss | val bpb | Δ loss | tokens/s |
|---|---|---|---|---|---|---|
| baseline | 12.39M | 1 | 1.8329 | 0.6490 | +0.0000 | 24,902 |
| mlp=swiglu | 12.39M | 1 | 1.8126 | 0.6418 | -0.0203 | 22,985 |
| n_kv_head=2 | 11.21M | 1 | 1.8468 | 0.6540 | +0.0139 | 26,202 |
| norm=rmsnorm | 12.39M | 1 | 1.8312 | 0.6484 | -0.0017 | 23,823 |
| pos_emb=rope | 12.19M | 1 | 1.7621 | 0.6240 | -0.0709 | 22,277 |
| n_kv_head=2, pos_emb=rope, norm=rmsnorm, mlp=swiglu ⚠ 4 changes | 11.02M | 1 | 1.7208 | 0.6093 | -0.1121 | 22,448 |

All runs: 99.9M tokens, block 512, lr 0.001, data_seed 1337, 2000 val windows, bfloat16, tokenizer tokenizer/tinystories_4k.json.
Groups with 1 seed have no error bar: a small Δ loss there may just be seed noise.
