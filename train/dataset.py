import os
import torch
from transformers import GPT2TokenizerFast
import json
import time
import requests
import base64
import torch.nn.functional as F


config = json.load(open('config.json'))
tokenizer = GPT2TokenizerFast.from_pretrained(config['tokenizer_name'])  # define tokenizer
chunk_size = config['chunk_size']   # chunk token length
max_length = config['max_length']   # model max_length
request_server = config['request_server']   # request_server ip

tokenizer.pad_token = tokenizer.eos_token
num_chunk = int(max_length / chunk_size)

headers = {
    'accept': 'application/json',
    # Already added when you pass json= but not when you pass data=
    # 'Content-Type': 'application/json',
}


def exist(path):
    return os.path.exists(path)


def split_token(text):
    encode_dict = tokenizer(text, return_offsets_mapping=True, max_length=max_length, truncation=True)
    input_ids = encode_dict['input_ids']
    attention_mask = encode_dict['attention_mask']
    offset_mapping = encode_dict['offset_mapping']
    chunk_count = int(len(offset_mapping) / chunk_size)
    chunk_texts = []
    for chunk_offset in range(chunk_count):
        chunk_mappings = offset_mapping[chunk_offset * chunk_size:chunk_offset * chunk_size + chunk_size]
        chunk_text = text[chunk_mappings[0][0]:chunk_mappings[-1][1]]
        chunk_texts.append(chunk_text)
    retrieval = None
    if len(input_ids) >= chunk_size:
        data = {"query": [base64.b64encode(s.encode('utf-8')).decode('utf-8') for s in chunk_texts]}
        while retrieval is None:
            try:
                #response = requests.post(request_server, headers=headers, json=data)
                #retrieval = json.loads(response.text)
                # Mock retrieval
                retrieval = []
                for i in range(len(chunk_texts)):
                    retrieval.append(['abc', 'abc'])
            except Exception as e:
                time.sleep(1)
                print('retrieval failed' + str(e))
    else:
        retrieval = []
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long), retrieval


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, data, pad_to_batch):
        with open(data, 'r') as f:
            self.data = f.readlines()
        self.pad_to_batch = pad_to_batch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        json_data = json.loads(self.data[index])
        text = json_data['text']
        ids, attention_mask, retrieval = split_token(text)
        labels = ids.clone()
        # prepare retrieval
        if retrieval == []:
            retrieval = None
        else:
            chunk_1 = [i[0] for i in retrieval]
            chunk_2 = [i[1] for i in retrieval]
            chunk_1 = torch.tensor(tokenizer(chunk_1, max_length=chunk_size * 2, padding="max_length", truncation=True).input_ids).unsqueeze(1)
            chunk_2 = torch.tensor(tokenizer(chunk_2, max_length=chunk_size * 2, padding="max_length", truncation=True).input_ids).unsqueeze(1)
            retrieval = torch.cat((chunk_1, chunk_2), dim=1)

        # pad to batch
        if self.pad_to_batch:
            if retrieval is None:
                retrieval = torch.ones(1, 2, chunk_size * 2)
            if ids.shape[0] < max_length:
                attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[0]), value=0)
                ids = F.pad(ids, (0, max_length - ids.shape[0]), value=tokenizer.pad_token_id)
                labels = F.pad(labels, (0, max_length - labels.shape[0]), value=-100)
            if retrieval.shape[0] != num_chunk:
                pad_ = torch.ones(num_chunk - retrieval.shape[0], retrieval.shape[1], retrieval.shape[2])
                pad_ = pad_ * tokenizer.pad_token_id
                retrieval = torch.cat((retrieval, pad_), dim=0)

        return {"input_ids": ids.long(), "attention_mask": attention_mask.long(), "labels": labels.long(), "retrieval": retrieval.long() if retrieval is not None else None}
