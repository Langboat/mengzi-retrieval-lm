import torch
from transformers import Re_gptForCausalLM, Trainer, TrainingArguments
import json
from dataset import RetrievalDataset
seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print("set seed")

config = json.load(open('config.json'))
deepspeed_config = json.load(open('fitting_dp.json'))

model_name = config['model_name']
output_dir = config['output_dir']
learning_rate = config['learning_rate']
num_train_epochs = config['num_train_epochs']
per_device_train_batch_size = config['per_device_train_batch_size']
per_device_eval_batch_size = config['per_device_eval_batch_size']
logging_dir = config['logging_dir']
logging_steps = config['logging_steps']
evaluation_strategy = config['evaluation_strategy']
save_strategy = config['save_strategy']
load_best_model_at_end = config['load_best_model_at_end']
do_train = config['do_train']
do_eval = config['do_eval']
no_cuda = config['no_cuda']
eval_steps = config['eval_steps']
gradient_accumulation_steps = config['gradient_accumulation_steps']
save_steps = config['save_steps']
save_total_limit = config['save_total_limit']
fp16 = config['fp16']
bf16 = config['bf16']
weight_decay = config['weight_decay']
adam_beta2 = config['adam_beta2']
warmup_ratio = config['warmup_ratio']
dataloader_num_workers = config['dataloader_num_workers']

retro = Re_gptForCausalLM.from_pretrained(model_name)
train_dataset = RetrievalDataset(config['train_dataset_path'], pad_to_batch=True)
test_dataset = RetrievalDataset(config['test_dataset_path'], pad_to_batch=False)

training_args = TrainingArguments(
    output_dir=output_dir,          # output directory
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,              # total number of training epochs
    per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=per_device_eval_batch_size,   # batch size for evaluation
    logging_dir=logging_dir,            # directory for storing logs
    logging_steps=logging_steps,
    evaluation_strategy=evaluation_strategy,
    save_strategy=save_strategy,
    load_best_model_at_end=load_best_model_at_end,
    do_train=do_train,
    do_eval=do_eval,
    no_cuda=no_cuda,
    eval_steps=eval_steps,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    deepspeed=deepspeed_config,
    fp16=fp16,
    bf16=bf16,
    weight_decay=weight_decay,
    adam_beta2=adam_beta2,
    warmup_ratio=warmup_ratio,
    dataloader_num_workers=dataloader_num_workers
)

print(training_args)

trainer = Trainer(
    model=retro,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,          # evaluation dataset
)
train_out = trainer.train()
