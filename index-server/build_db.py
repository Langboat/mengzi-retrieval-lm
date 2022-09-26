import argparse
import ray
import faiss 
import lmdb
import json
import pickle
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--rank', type=int, required=True)
args = parser.parse_args()
rank = args.rank

env = lmdb.open(f'db/{rank}', map_size=1024*1024*1024*10)

print('Loading data...')
ds = ray.data.read_parquet(f'./shards/{rank}.parquet')
print('Data loaded!')
print('Data count:', ds.count())

print('Writing data to lmdb...')
count = 0
txn = env.begin(write=True)
for row in tqdm(ds.iter_rows(), total=ds.count()):
    txn.put(
        str(count).encode(),
        pickle.dumps({
            'text': row['text'], 
            'next_chunk': row['next_chunk']
        })
    )
    count += 1
txn.commit()
env.close()
print('Data written!')

