# How to build with your own data
1. Prepare your corpus and split it into small jsonl files with the following structure (like Pile, but we only need the `text` key), and put them into a folder named `splits`.
```
{
  'meta': {'pile_set_name': 'Pile-CC'},
  'text': 'It is done, and submitted. You can play “Survival of the Tastiest” on Android, and on the web. Playing on...'
}
```
2. Build metadata
```bash
python build_split_meta.py 
```
3. Build shard
```bash
python build_shard.py
```
4. Build database
```bash
build_db.py
```
5. Build index. The Faiss parameter is hardcoded now. Choosing an index is like a kind of compute resource and recall tradeoff. See [Faiss wiki](https://github.com/facebookresearch/faiss/wiki/The-index-factory) for more details.
```bash
build_index.py
```
