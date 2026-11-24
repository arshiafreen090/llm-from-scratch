| changes vs baseline | params | seeds | val loss | val bpb | Δ loss | tokens/s |
|---|---|---|---|---|---|---|
| baseline | 12.39M | 2 | 1.8322 ± 0.0010 | 0.6488 ± 0.0004 | +0.0000 | 24,952 |
| mlp=swiglu | 12.39M | 2 | 1.8062 ± 0.0090 | 0.6396 ± 0.0032 | -0.0260 | 23,010 |
| n_kv_head=2 | 11.21M | 2 | 1.8480 ± 0.0017 | 0.6544 ± 0.0006 | +0.0158 | 26,209 |
| norm=rmsnorm | 12.39M | 2 | 1.8322 ± 0.0014 | 0.6488 ± 0.0005 | +0.0000 | 23,826 |
| pos_emb=rope | 12.19M | 2 | 1.7536 ± 0.0120 | 0.6209 ± 0.0043 | -0.0786 | 22,331 |
| n_kv_head=2, pos_emb=rope, norm=rmsnorm, mlp=swiglu ⚠ 4 changes | 11.02M | 2 | 1.7254 ± 0.0065 | 0.6110 ± 0.0023 | -0.1068 | 22,451 |

All runs: 99.9M tokens, block 512, lr 0.001, data_seed 1337, 2000 val windows, bfloat16, tokenizer tokenizer/tinystories_4k.json.
