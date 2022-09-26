import json
import argparse
from transformers import GPT2TokenizerFast
import requests
import time
import base64
from tqdm import tqdm
from multiprocessing import Pool

config = json.load(open('config.json'))
tokenizer = GPT2TokenizerFast.from_pretrained(config['tokenizer_name'])
chunk_size = config['chunk_size']
max_length = config['max_length']

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--retrievaled_path', type=str)
args = parser.parse_args()
data_path = args.data_path
retrievaled_path = args.retrievaled_path
request_server = config['request_server']
headers = {
    'accept': 'application/json',
    # Already added when you pass json= but not when you pass data=
    # 'Content-Type': 'application/json',
}


def range_chunked(max_value, *, batch_size):
    counter = 0
    while counter < max_value:
        curr = counter + batch_size
        curr = min(curr, max_value)
        yield slice(counter, curr)
        counter = curr


def split_token(text):
    encode_dict = tokenizer(text, return_offsets_mapping=True, max_length=max_length, truncation=True)
    input_ids = encode_dict['input_ids']
    offset_mapping = encode_dict['offset_mapping']
    chunk_count = int(len(offset_mapping) / chunk_size)
    chunk_texts = []
    for chunk_offset in range(chunk_count):
        chunk_mappings = offset_mapping[chunk_offset * chunk_size:chunk_offset * chunk_size + chunk_size]
        chunk_text = text[chunk_mappings[0][0]: chunk_mappings[-1][1]]
        chunk_texts.append(chunk_text)
    return chunk_texts, chunk_count, input_ids, text


def process_indice(indice):
    indice = f_data[indice]
    all_chunk_texts = []
    all_chunk_count = {}
    all_input_ids = []
    all_text = []
    for i in range(len(indice)):
        chunk_texts, chunk_count, input_ids, text = split_token(indice[i])
        all_chunk_texts.append(chunk_texts)
        all_chunk_count[i] = chunk_count
        all_input_ids.append(input_ids)
        all_text.append(text)
    return all_chunk_texts, all_chunk_count, all_input_ids, all_text


def find_retrieval(chunk_texts):
    retrieval = []
    data = {"query": [base64.b64encode(s.encode('utf-8')).decode('utf-8') for s in chunk_texts]}
    if chunk_texts != []:
        while retrieval == []:
            try:
                response = requests.post(request_server, headers=headers, json=data)
                retrieval = json.loads(response.text)
            except Exception as e:
                time.sleep(1)
                print("retrieval failed", e)
    else:
        return []
    return retrieval


def get_json(i):
    all_chunk_texts, all_chunk_count, all_input_ids, all_text = process_indice(i)
    flat_all_chunk_texts = []
    all_json_data = []
    if all_chunk_texts != []:
        flat_all_chunk_texts = [item for sublist in all_chunk_texts for item in sublist]
        retrieval = find_retrieval(flat_all_chunk_texts)
        assert len(retrieval) == len(flat_all_chunk_texts)
        start = 0
        end = 0
        for j in range(len(all_input_ids)):
            json_data = {}
            json_data['input_ids'] = all_input_ids[j]
            json_data['retrieval'] = retrieval[start: end + all_chunk_count[j]]
            json_data['text'] = all_text[j]
            assert len(json_data['retrieval']) == all_chunk_count[j]
            end = end + all_chunk_count[j]
            start = end
            all_json_data.append(json_data)
        assert end == len(retrieval)
    else:
        for j in range(len(all_input_ids)):
            json_data = {}
            json_data['input_ids'] = all_input_ids[j]
            json_data['text'] = all_text[j]
            json_data['retrieval'] = None
            all_json_data.append(json_data)
    return all_json_data


with open(data_path) as f:
    f_data = f.readlines()
f_data = [json.loads(x)['text'] for x in f_data]
all_indice = []
for i in range_chunked(len(f_data), batch_size=64):
    all_indice.append(i)
print("start")
with open(retrievaled_path, 'w') as f:
    with Pool(processes=64) as p:
        data = list(tqdm(p.imap(get_json, all_indice), total=len(all_indice)))
    flat_data = [item for sublist in data for item in sublist]
    for x in tqdm(flat_data):
        f.write(json.dumps(x) + '\n')
