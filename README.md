# Retrieval-LM

"Retrieval" is an important way to improve the performance of language models. This repository is an experimental implementation of the retrieval-enhanced language model. **Currently, it only supports retrieval fitting on GPT-Neo.**

We forked [Huggingface Transformers](https://github.com/huggingface/transformers) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to add retrieval support. The indexing part is implemented as an HTTP server to decouple retrieval and training better.

Most of the model implementation is copied from
[RETRO-pytorch](https://github.com/lucidrains/RETRO-pytorch) and [GPT-Neo](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py). We use `transformers-cli` to add a new model named `Re_gptForCausalLM` based on GPT-Neo, and then add retrieval part to it.

You can initialize a model like this:
```python
from transformers import Re_gptForCausalLM
model = Re_gptForCausalLM.from_pretrained(model_name)
```
And evaluate the model like this:
```bash
python main.py \
    --model retrieval \
    --model_args pretrained=model_path \
    --device 0 \
    --tasks wikitext,lambada,winogrande,mathqa,pubmedqa  \
    --batch_size 1
```

# Architecture

![Cloud Architecture - Page 1 (1)](https://user-images.githubusercontent.com/1523477/193192744-6544da36-c281-41cc-8199-e6dde456be3b.png)


# Usage

## Environment
TODO: how to setup environment with conda

## Download index and model
TODO: how to download index and model from huggingface model hub

## Setup retrieval server
TODO: how to setup retrieval server with pre-downloaded data

## Training
TODO: example code

## Inference
TODO: example code

# Evaluations
TODO: example code

# Citations
TODO:
