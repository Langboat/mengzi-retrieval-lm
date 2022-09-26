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

print('Loading data...')
ds = ray.data.read_parquet(f'./shards/{rank}.parquet')
print('Data loaded!')
print('Data count:', ds.count())

print('Building index...')
res = faiss.StandardGpuResources()
res.setTempMemory(1024 * 1024 * 64)
co = faiss.GpuClonerOptions()
co.useFloat16 = True
cpu_index = faiss.index_factory(384, 'IVF1024,PQ64')
index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
#index = faiss.IndexFlatL2(384)
train_data = ds.take(1048576)
xt = np.stack([x['embedding'] for x in train_data]).astype('float32')
print('Training data loaded!')
index.train(xt)
print('Training finished!')

bs = 4096
for batch in tqdm(ds.iter_batches(batch_size=bs), total=ds.count()//bs):
    xb = np.stack(batch.embedding.values).astype('float32')
    index.add(xb)
print('Index built!')
    
faiss.write_index(faiss.index_gpu_to_cpu(index), f'./indexes_IVF1024PQ64/{rank}')
print('Index written!')
