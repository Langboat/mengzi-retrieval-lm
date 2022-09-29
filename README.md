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
conda create -n mengzi-retrieval-fit python=3.7
conda activate mengzi-retrieval-fit
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
git clone git@github.com:bling0830/mengzi-retrieval-lm.git
cd mengzi-retrieval-lm
git submodule update --init --recursive
cd transformers/
pip install -e .
pip install -r requirement.txt
```

## Download index and model
### Download index
Using IVF1024PQ48 as the faiss indexfactory, we submitted the index and database, which can be downloaded using the following command. 

In download_index_db.py, you can specify the number of indexes and databases you want to download.
```bash
python -u download_index_db.py
```


## Setup retrieval server
## setup
To begin the retrieval service, develop the distributed retrieval service using the ray framework, and launch the retrieval service using api.py.

You can initialize a retrieval service like this:
```bash
cd index-server/
ray start --head
python -u api.py \
--config config_IVF1024PQ48.json \
--db_path 
```
`Keep in mind that the config IVF1024PQ48.json shard count must match the number of downloaded indexes.`


· db_path：the database's download location from huggingface. 
"../db/models—Langboat—Pile-DB/snapshots/fd35bcce75db5c1b7385a28018029f7465b4e966" is an example.  

This command will download the database and index data from huggingface. 

Change the index folder in the configuration file (config IVF1024PQ48) to point to the index folder's path, and send the database folder's snapshots as the db path to the api.py script.

## stop
Stop the retrieval service with the following command
```bash
ray stop
```
## Training
Use train/train.py to implement training; train/config.json can be modified to change the training parameters.

You can initialize training like this:
```bash
cd train
python -u train.py
```

## Inference
Utilize train/inference.py as an inference to determine the text's loss and perplexity.
```bash
cd train
python -u inference.py \
    --model_path \
    --file_name 
```

# Evaluations
Use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) as evaluation method
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
· model_path：the fitting model path
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
