import torch
import json
import torch.nn.functional as F
from einops import rearrange
from transformers import Re_gptForCausalLM, GPT2TokenizerFast
import argparse
from tqdm import tqdm
from dataset import split_token
seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--text', type=str, help="input text, if you want to use retrieval, tokenized input text must larger than 64 token", default="""Q:\n\n\u00bfPorqu\u00e9 en este loop de JavaScript la impresi\u00f3n de la variable es desde counter y no desde counter-1?\n\nen mi b\u00fasqueda por aprender programaci\u00f3n por mis propios medios, me he topado con el tema de recursividad y este simple c\u00f3digo... mi pregunta ya que la variable counter comienza desde 10 y dentro del loop While el contador resta 1, porqu\u00e9 en la \"impresi\u00f3n\" aparece desde el 10. S\u00e9 que si quisiera empezar desde 10 colocar\u00eda el contador en 11... pero obviamente tengo la curiosidad y no entiendo.\nvar counter = 10;\nwhile(counter > 0) {\n    console.log(counter--);\n}\n\nresultado:\n10\n9\n8\n7\n6\n5\n4\n3\n2\n1\n\nA:\n\nLa raz\u00f3n es simple, en recursividad lo que haces es pasar una variable o arreglo en la mayor parte de los caso para modificarlos o simplemente imprimirlos, en tu caso quieres restar un numero por cada iteracion dentro de tu ciclo while pero aqui lo que tu quieres conseguir es que primero te imprima el 9 por la l\u00f3gica que encuentras en tu programa y aunque no es del todo err\u00f3nea eso no suceder\u00e1 jamas por la siguiente raz\u00f3n.\nEn tu codigo lo que tienes es la impresion de tu variable e imprimes lo que es counter-- y a pesar de que si te resta -1 en esa misma iteracion sucede que primero te imprimira la variable antes de hacer dicha operacion ya que es lo que primero lee javascript, es como si tu codigo estuviera dividido en dos partes.\nEJEMPLO\nvar counter = 10;\nwhile(counter > 0) {\n    console.log(counter); // Lee antes el valor variable\n    counter--; // Despu\u00e9s realiza operaci\u00f3n\n}\n\nEsto sucede asi porque es como funciona internamente lo que realizas con javascript ya que a pesar de que parece un metodo simple de resta internamente esta compuesto de dos partes. Para cuando javascript hace la operacion tu valor ya esta en pantalla.\nEJEMPLO VISUAL\n\nPrimera iteraci\u00f3n:\ncounter = 10 | counter-- | counter = 9\ncounter = 9 | counter-- | counter = 8\ncounter = 8 | counter-- | counter = 7\n...\ncounter = 1 | counter-- | counter = 0\ncounter = 0 | counter-- | counter = -1 -> En este caso ya no cumples con la condici\u00f3n por lo cual nunca se imprime.\n\nPara realizar el proceso que quieres en el caso de que primero quieras que se imprima el 9 entonces deberas de hacer lo siguiente:\n\nvar counter = 10;\nwhile(counter > 0) {\n    counter--;\n    console.log(counter);\n}\n.as-console-wrapper { max-height: 100% !important; top: 0; }\n\n""")
parser.add_argument('--model_path', type=str, help='fitting model path')
parser.add_argument('--retrieval', type=list, default=None, help='if you want to set retrieval, you should input retrieval list, length of retrieval list should be (input length/chunk_size), if not set retrieval, retrieval will be search online')
parser.add_argument('--temperature', type=float, default=1., help='temperature for gumbel softmax')
parser.add_argument('--top_k', type=float, default=0.9, help='top_k for gumbel softmax')
parser.add_argument('--top_p', type=float, default=0.9, help='top_p for gumbel softmax')
args = parser.parse_args()

config = json.load(open('config.json'))


def exists(val):
    return val is not None


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


def safe_cat(accum, t, dim=-1):
    if not exists(accum):
        return t
    return torch.cat((accum, t), dim=dim)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def tokenize_retrieval(retrieval):
    chunk_1 = [i[0] for i in retrieval]
    chunk_2 = [i[1] for i in retrieval]
    chunk_1 = torch.tensor(tokenizer(chunk_1, max_length=chunk_size * 2, padding="max_length", truncation=True).input_ids).unsqueeze(1)
    chunk_2 = torch.tensor(tokenizer(chunk_2, max_length=chunk_size * 2, padding="max_length", truncation=True).input_ids).unsqueeze(1)
    retrieval = torch.cat((chunk_1, chunk_2), dim=1)
    return retrieval


def fetch_knn_chunks_fn(chunk):
    text = tokenizer.decode(chunk[0])
    ids, attention_mask, retrieval = split_token(text)
    retrieval = tokenize_retrieval(retrieval)
    return retrieval


def generate_to_maxseq(
        retro,
        seq,
        max_seq_len,
        retrieval=None,
        filter_fn=top_k,
        filter_thres=1.0,
        temperature=1.0,
        chunk_size=config['chunk_size']):
    assert filter_fn in {top_k, top_p}, 'filter function must be either top-k or nucleus'

    device = next(retro.parameters()).device

    b, start_seq_len = seq.shape

    # move onto same device as RETRO

    seq = seq.to(device)

    # prepare retrieval related variables

    # get starting sequence index

    out = seq

    # sampling loop

    for i in tqdm(range(start_seq_len - 1, max_seq_len)):

        logits = retro(out, retrieval=retrieval).logits
        logits = logits[:, i]

        logits = filter_fn(logits, thres=filter_thres)
        sampled = gumbel_sample(logits, temperature=temperature, dim=-1)
        sampled = rearrange(sampled, 'b -> b 1')

        out = torch.cat((out, sampled), dim=1)

        # early terminate if all EOS

        is_eos_tokens = (out == EOS_ID)

        if is_eos_tokens.any(dim=-1).all():

            # mask out everything after the eos tokens

            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
            out = out.masked_fill(mask, retro.pad_id)
            break

        # when the sequence length is a multiple of the chunk size
        # retrieve the next set of knns

        curr_seq_len = out.shape[-1]

        if (curr_seq_len % chunk_size) == 0 and retrieval is not None:
            last_chunk = rearrange(out, 'b (c n) -> b c n', n=chunk_size)[:, -1]

            knn_chunks = fetch_knn_chunks_fn(last_chunk)

            # concat retrieval knn chunks to all retrieval
            # to be sent to Retro for chunked cross attention at the next iteration

            knn_chunks = rearrange(knn_chunks, 'b k r -> b 1 k r')
            retrieval = safe_cat(retrieval, knn_chunks, dim=1)

            print(f'retrieval at {curr_seq_len} / {max_seq_len}')
    return out


retro = Re_gptForCausalLM.from_pretrained(args.model_path)
retro.eval()
tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-125M")
EOS_ID = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
text = args.text
retrieval_set = args.retrieval
chunk_size = config['chunk_size']
ids, attention_mask, retrieval = split_token(text)
retrieval = tokenize_retrieval(retrieval)
if retrieval_set is not None and ids.shape[0] > chunk_size:
    assert len(retrieval_set) == len(retrieval), 'retrieval set and retrieval must be the same length'
    retrieval = retrieval_set
ids = ids.unsqueeze(0)
retrieval = retrieval.unsqueeze(0)

print("Generating...")
out = generate_to_maxseq(retro, ids, retrieval=retrieval, max_seq_len=config['max_length'])
print(tokenizer.decode(out[0, :].tolist()))
