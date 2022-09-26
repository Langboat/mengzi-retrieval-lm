import argparse
import math
import os
import pandas as pd
import ray
from ray.data.impl.compute import ActorPoolStrategy
import json
from transformers import GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('--rank', type=int, required=True)
args = parser.parse_args()
rank = args.rank

split_meta_data = pd.read_json('./metadata/split_meta_info.jsonl', orient='records', lines=True)
split_meta_data = split_meta_data[split_meta_data['shard_id'] == rank]

print("Shard ID: ", rank)
print("Loading files...")
ds = ray.data.read_text(split_meta_data['file_name'].values.tolist())
ds = ds.map(lambda s: json.loads(s))
print("Files loaded.")

class SplitChunk:
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained('EleutherAI/gpt-neo-125M')
    
    def process_line(self, line: str):
        encode_dict = self.tokenizer(line, return_offsets_mapping=True)

        input_ids = encode_dict['input_ids']
        offset_mapping = encode_dict['offset_mapping']

        assert len(input_ids) == len(offset_mapping)

        chunk_count = math.ceil(len(offset_mapping)/64)
        chunk_data = []
        for chunk_offset in range(chunk_count):
            chunk_mappings = offset_mapping[chunk_offset*64:chunk_offset*64+64]
            chunk_text = line[chunk_mappings[0][0]:chunk_mappings[-1][1]]
            chunk_data.append({
                'text': chunk_text,
                'next_chunk': None
            })
        for i in range(len(chunk_data)-1):
            chunk_data[i]['next_chunk'] = chunk_data[i+1]['text']
        return chunk_data
                
    
    def __call__(self, batch: pd.DataFrame):
        data = []
        for line in batch['text'].values:
            data.extend(self.process_line(line))
        return pd.DataFrame(data)
        
print("Split chunks...")
ds = ds.map_batches(SplitChunk, compute=ActorPoolStrategy(2, 16))
print("Split chunks done.")

class EncodeChunk:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L12-v2')
    
    def __call__(self, batch: pd.DataFrame):
        batch['embedding'] = self.model.encode(batch['text'].values).tolist()
        return batch
    
print("Encode chunks...")
ds = ds.map_batches(EncodeChunk, compute=ActorPoolStrategy(2, 10), num_gpus=0.1)
print("Encode chunks done.")

print("Save shard...")
ds.write_parquet(f'shards/{rank}.parquet')
print("Done.")
