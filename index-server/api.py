import uvicorn
import argparse
import json
import random
import hashlib
import base64
import pickle
import time
import ray
import faiss
import lmdb
from sentence_transformers import SentenceTransformer
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import zlib

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='config file')
args = parser.parse_args()
config = json.load(open(args.config))
print(config)

ray.init(address='127.0.0.1:6379')

@ray.remote(num_cpus=config['index_num_cpus'], num_gpus=config['index_num_gpus'])
class IndexActor(object):
    def __init__(self, rank):
        if config['index_num_gpus'] == 0:
            self.index = faiss.read_index(f'{config["index_folder"]}/{rank}')
        else:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024*1024*256)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            cpu_index = faiss.read_index(f'{config["index_folder"]}/{rank}')
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
        self.index.nprobe = config['index_nprobe']
        env = lmdb.open(f'./db/{rank}', readonly=True, readahead=True, map_size=1024*1024*1024*8)
        # Get all records from the database
        self.records = {}
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                item = pickle.loads(value)
                text = item['text']+str(item['next_chunk'] or '')
                ztext = zlib.compress(text.encode('utf-8'))
                self.records[key.decode('utf-8')] = ztext
        env.close()
        print(f'{rank} index actor initialized')
    
    async def search(self, xq, k=2):
        search_start = time.time()
        distances, ids = self.index.search(xq, k)
        faiss_time = time.time() - search_start
        if faiss_time > 2:
            print(f'faiss_time: {faiss_time}')
        all_results = []
        for query_idx in range(xq.shape[0]):
            query_results = []
            for neighbor_idx in range(k):
                ztext = self.records[str(ids[query_idx][neighbor_idx])]
                text = zlib.decompress(ztext).decode('utf-8')
                query_results.append((
                    distances[query_idx][neighbor_idx],
                    text
                ))
            all_results.append(query_results)
        search_time = time.time() - search_start
        if search_time > 3:
            print(f'search_time: {search_time}')
        return all_results


@ray.remote(num_cpus=config['encoder_num_cpus'], num_gpus=config['encoder_num_gpus'])
class EncoderActor(object):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L12-v2')

    async def encode(self, sentences):
        return self.model.encode(sentences)

index_actor_count = config['shard_count']
encoder_actor_count = config['encoder_actor_count']
init_cmds = [f'index:{i}' for i in range(index_actor_count)] + ['encoder'] * encoder_actor_count
random.shuffle(init_cmds)
index_actors = []
encoder_actors = []
for cmd in init_cmds:
    if cmd.startswith('index'):
        index_actors.append(IndexActor.remote(int(cmd[6:])))
    else:
        encoder_actors.append(EncoderActor.remote())

cache = {}

async def retrieval(querys):
    start = time.time()
    encoder_actor = random.choice(encoder_actors)
    embs = await encoder_actor.encode.remote(querys)
    embedding_time = time.time() - start
    if embedding_time > 1:
        print(f'embedding_time: {embedding_time}')
    gather_start = time.time()
    results = await asyncio.gather(*[a.search.remote(embs) for a in index_actors])
    gather_time = time.time() - gather_start
    if gather_time > 3:
        print(f'gather_time: {gather_time}')
    
    all_results = []
    for query_idx in range(len(querys)):
        query_results = []
        for rank in range(len(index_actors)):
            query_results.extend(results[rank][query_idx])
        query_results = sorted(query_results, key=lambda x: x[0])
        query_results = [x[1] for x in query_results][:2]
        all_results.append(query_results)
    duration = time.time() - start
    if duration > 1:
        print(f'Too slow: {duration:.2f}s')

    return all_results

app = FastAPI()

class Item(BaseModel):
    query: list

@app.post("/")
async def search(item: Item):
    query_list = [base64.b64decode(q).decode('utf-8') for q in item.query]

    # Use in-memory cache
    cached_results = []
    uncached_querys = []
    for i in range(len(query_list)):
        if query_list[i] in cache.keys():
            cached_results.append(cache[query_list[i]])
        else:
            cached_results.append(None)
            uncached_querys.append(query_list[i])

    print(f'{len(uncached_querys)}/{len(query_list)} queries are uncached')
    if len(uncached_querys) > 0:
        uncached_results = await retrieval(uncached_querys)
        for q, r in zip(uncached_querys, uncached_results):
            cache[q] = r

    # Combine cached and uncached results
    all_results = []
    for i in range(len(query_list)):
        if cached_results[i] is None:
            all_results.append(uncached_results.pop(0))
        else:
            all_results.append(cached_results[i])
    return all_results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
