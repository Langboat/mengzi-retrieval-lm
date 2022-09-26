from tqdm import tqdm
import math
import argparse
from transformers import Re_gptForCausalLM, GPT2TokenizerFast
from dataset import RetrievalDataset
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='fitting model path')
parser.add_argument('--file_name', type=str, help='inference file name, calculate the average loss and perplexity')
args = parser.parse_args()
tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
model_path = args.model_path
model = Re_gptForCausalLM.from_pretrained(model_path)
model.eval()
file_name = args.file_name
ppl = []
all_loss = []
dataset = RetrievalDataset(file_name, pad_to_batch=False)
for i in tqdm(dataset):
    ids = i['input_ids'].unsqueeze(0)
    labels = i['labels'].unsqueeze(0)
    attention_mask = i['attention_mask'].unsqueeze(0)
    retrieval = i['retrieval'].unsqueeze(0) if i['retrieval'] is not None else None
    loss = model(input_ids=ids, labels=labels, attention_mask=attention_mask, retrieval=retrieval).loss
    all_loss.append(loss.item())
    ppl.append(math.exp(loss.item()))
print(sum(all_loss) / len(all_loss))
print(sum(ppl) / len(ppl))
