import os
from glob import glob
import pandas as pd

SHARD_MAX_SIZE = 1024*1024*1024*1

file_names = glob('splits/*/*', recursive=True)
file_names.sort()

shard_ids = []
current_shard_id = 0
current_shard_size = 0

for file_name in file_names:
    file_size = os.path.getsize(file_name)
    if current_shard_size + file_size > SHARD_MAX_SIZE:
        current_shard_id += 1
        current_shard_size = 0
    shard_ids.append(current_shard_id)
    current_shard_size += file_size

splited_files = pd.DataFrame({'file_name': file_names, 'shard_id': shard_ids})
splited_files.to_json('metadata/split_meta_info.jsonl', orient='records', lines=True)
