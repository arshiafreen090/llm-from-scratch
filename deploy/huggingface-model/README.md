---
language: en
pipeline_tag: text-generation
tags:
  - llama
  - from-scratch
  - tinystories
  - pytorch
datasets:
  - roneneldan/TinyStories
  - roneneldan/TinyStoriesInstruct
---

# TinyLLaMA from scratch

An 11M-parameter LLaMA-style language model (RoPE, RMSNorm, SwiGLU, grouped-query attention),
pretrained on TinyStories and instruction-tuned on TinyStories-Instruct. Every part was written
from scratch in PyTorch: the BPE tokenizer, the model, training, and inference. This is not a
Hugging Face `transformers` model.

- **Try it:** [chat with it in your browser (Space)](https://huggingface.co/spaces/afeening/tinyllama): the
  ONNX file below runs on your own device, so nothing is sent to a server
- **Code, training runs and results:** https://github.com/arshiafreen090/llm-from-scratch

## Files

| file | what it is | size |
|---|---|---|
| **`sft-small-llama-s1-v2-int8.pt`** | **the model to use**: instruction-tuned (v2) on 300k examples with varied instruction phrasings, INT8 weights | 11 MB |
| `sft-small-llama-s1-v2-int8.onnx` (+ `.onnx.json`) | v2 as ONNX, with the KV cache as inputs/outputs: what the browser demo runs (ONNX Runtime Web). Checked to give token-for-token the same greedy output as PyTorch | 11 MB |
| `sft-small-llama-s1-int8.pt` | v1: instruction-tuned on 100k examples with 45 fixed instruction templates; follows requests only when worded like them | 11 MB |
| `small-llama-s1-int8.pt` | pretrained model, INT8 weights: continues a story | 11 MB |
| `small-llama-s1.pt` | the same pretrained model in float32: the starting point for fine-tuning (`python -m train.sft --init_ckpt`) | 44 MB |
| `draft-llama.pt` | 2.5M-parameter draft model for speculative decoding (float32) | 10 MB |
| `sft-small-llama-s1-int8.onnx` (+ `.onnx.json`) | v1 as ONNX | 11 MB |
| `tinystories_4k.json` | the 4,096-token BPE tokenizer all of them use | |

The files hold only tensors and plain settings, so they load with PyTorch's safe
`weights_only=True` loader.

## Results

- **Architecture study:** each LLaMA change was tested against GPT-2 separately and all together, under
  identical conditions (2 seeds, 100M tokens). The full model beats the GPT-2 baseline by 0.107 validation
  loss with 11% fewer parameters, and reaches the baseline's final loss with 62% of the tokens.
- **Instruction following:** the pretrained model uses 4% of the words an instruction asks for (chance).
  v1 (100k examples, 45 fixed instruction templates) reached 51% on template-phrased requests but 39% on
  varied ones. **v2** (300k examples, instructions phrased the way people type) uses **62%** of the required
  words and **53%** of the plot's words on varied requests, including phrasings held out of training.
- **INT8:** 4× smaller than float32, with validation loss changed by 0.0001.

## Use

The code that runs the model lives in the GitHub repo:

```bash
git clone https://github.com/arshiafreen090/llm-from-scratch.git && cd llm-from-scratch
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-serve.txt
hf download afeening/tinyllama-from-scratch sft-small-llama-s1-v2-int8.pt --local-dir checkpoints
python -m serve.api --ckpt checkpoints/sft-small-llama-s1-v2-int8.pt      # chat UI at http://localhost:8000
python -m inference.generate --ckpt checkpoints/sft-small-llama-s1-v2-int8.pt --chat \
       --prompt "write a story about a fish dancing in a river. use the words rain and happy"
```

## Limitations

The model writes simple children's stories in the TinyStories style and nothing else. It follows
instructions only partly: it uses about half of the requested words, and all of them in about 13% of
stories. It answers one instruction at a time, with no memory of earlier messages. It's a learning
project, not an assistant.
