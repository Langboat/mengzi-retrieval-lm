from huggingface_hub import hf_hub_download
from multiprocessing import Pool


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
    p.map(download_file, range(200))