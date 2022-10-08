from huggingface_hub import hf_hub_download
from multiprocessing import Pool
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num', type=int, help='the number of indexes and databases you want to download')
args = parser.parse_args()


def download_file(i):
    i = str(i)
    hf_hub_download(repo_id="Langboat/Pile-index-IVF1024PQ48",
        filename=i,
        cache_dir="index/",
        force_download=False)
    hf_hub_download(repo_id="Langboat/Pile-DB",
        subfolder=i,
        filename='lock.mdb',
        cache_dir="db/",
        force_download=False)
    hf_hub_download(repo_id="Langboat/Pile-DB",
        subfolder=i,
        filename='data.mdb',
        cache_dir="db/",
        force_download=False)


with Pool(16) as p:
    p.map(download_file, range(args.num))