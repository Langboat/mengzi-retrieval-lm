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

# Usage

## Environment
```bash
git clone git@github.com:bling0830/mengzi-retrieval-lm.git
cd mengzi-retrieval-lm
git submodule update --init --recursive
cd transformers/
pip install -e .
```

## Download index and model
### Download index
```bash
python -u download_index_db.py
```


## Setup retrieval server
### setup
```bash
cd index-server/
ray start --head
python -u api.py \
--config config_IVF1024PQ48.json \
--db_path ../db/models--Langboat--Pile-DB/snapshots/fd35bcce75db5c1b7385a28018029f7465b4e966
```
This command will download the db data and index data from huggingface, modify the index_folder in config_IVF1024PQ48 to the path in the index folder, and pass the snapshots in the db folder as db_path to api.py

### stop
```bash
ray stop
```
## Training
```bash
cd train
python -u train.py
```

## Inference
```bash
cd train
python -u inference.py \
    --model_path \
    --file_name 
```

# Evaluations
```bash
cd lm-evaluation-harness
python setup.py install
```
### with retrieval
```bash
python main.py \
    --model retrieval \
    --model_args pretrained=model_path \
    --device 0 \
    --tasks wikitext  \
    --batch_size 1
```
### without retrieval
```bash
python main.py \
	--model gpt2 \
	--model_args pretrained=EleutherAI/gpt-neo-125M \
	--device 0 \
	--tasks wikitext \
	--batch_size 1
```
# Citations
TODO:
