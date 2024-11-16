import time

dataset = 'openwebtext'
init_from = 'gpt2'

out_dir = f'out-openwebtext-{init_from}'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'openwebtext'
wandb_run_name = 'ft-' + str(time.time())

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
batch_size = 8
gradient_accumulation_steps = 32
max_iters = 100

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False


# device = "mps"
compile = False
compute_grad_memory = True