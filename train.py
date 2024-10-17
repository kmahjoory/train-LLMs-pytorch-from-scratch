"""
Training can run on Cuda or CPU.

The followig code will run on GPU if available:

This will use the default config:
python train.py

or create a .yaml config file and store in config folder.  
python train.py --config_file train_config.yaml
"""

import os
import time
import math
import inspect
from datetime import datetime
import yaml
from types import SimpleNamespace
# import wandb # imported below if needed
import argparse
from dataclasses import dataclass
import tiktoken
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models import GPT
from src.dataloader import DataLoaderLite
from src.hellaswag import render_example, iterate_examples
from src.utils import get_most_likely_row, CosineScheduler, init_from_checkpoint

# Define the default configuration file path
CONFIG_FILE = "default_config.yaml"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train with a specific configuration.")
parser.add_argument("--config_file", type=str, default=None, help="Path to the YAML configuration file.")
args = parser.parse_args()
if args.config_file:
    CONFIG_FILE = args.config_file

with open("config/" + CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

# Enable logging (W&B)
if config["enable_wandb_logging"]:
    import wandb
    wandb.init(project=config["wandb_project_name"], 
    config=config, 
    dir=config["logs_dir"])
    config = wandb.config
config = SimpleNamespace(**config)

# encoder for text generation while training
enc = tiktoken.get_encoding("gpt2")

device_type = config.device_type
device = torch.device(device_type)
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ddp_rank = 0
master_process = True

checkpoint_dir = config.checkpoint_dir
os.makedirs(checkpoint_dir, exist_ok=True)
logs_dir = config.logs_dir
os.makedirs(logs_dir, exist_ok=True)

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Batching and Gradient accumulation
total_batch_size = config.total_batch_size # Total tokens per batch (2**19 ~ 0.5M tokens)
B = config.micro_batch_size # micro batch size
T = config.sequence_length 
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# Dataloader instantiaotion
train_loader = DataLoaderLite(B=B, T=T, split="train")
val_loader = DataLoaderLite(B=B, T=T, split="val")

# use TF32 or if available BF32 for linear matrix multiplications
torch.set_float32_matmul_precision('high')

# Instantiate model object
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
first_step = 0 # defaults to zero for training from scratch
model = GPT(config)
if config.init_model_from == 'checkpoint':
    checkpoint_path = os.path.join(config.checkpoint_dir, config.checkpoint_file)
    model, iter_num = init_from_checkpoint(model, checkpoint_path, device_type)
    first_step += iter_num

model.to(device)
model = torch.compile(model)

# Instantiate a cosine scheduler
max_lr = config.max_lr
min_lr = config.min_lr
warmup_steps = config.warmup_iters
max_steps = config.max_iters # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
lr_scheduler = CosineScheduler(min_lr, max_lr, warmup_steps, max_steps)

# Initialize a GradScaler. If enabled=False scaler is a no-op
#scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)


# START TRAINING ...
for _s in range(max_steps):
    
    step = _s + first_step
    t0 = time.time()
    last_step = (step == max_steps - first_step - 1)

    # Evaluate once in a while 
    if (step > 0 and config.evaluation_interval and step % config.evaluation_interval == 0) or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=dtype):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        # W&B logging after every evaluation
        if config.enable_wandb_logging:
            wandb.log({
                "iter": step,
                "train/loss": loss_accum.item(),
                "val/loss": val_loss_accum.item(),
                "lr": lr,      
            })
        print(f"validation loss: {val_loss_accum.item():.4f}")

        # Saving checkpoints
        if step > 0 and (step % config.checkpoint_interval == 0 or last_step):
            # optionally write model checkpoints
            now = datetime.now()
            formatted_date = now.strftime("%Y-%m-%d")
            checkpoint_path = os.path.join(checkpoint_dir, f"model_{step:05d}_{formatted_date}.pt")
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': model.config,
                'iter_num': step,
                'val_loss': val_loss_accum.item()
                }
            # you might also want to add optimizer.state_dict() and
            # rng seeds etc., if you wanted to more exactly resume training
            torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step > 0 and config.evaluation_interval and step % config.evaluation_interval == 0) or last_step:
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=dtype):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")


    # once in a while generate from the model (except step 0, which is noise)
    if (step > 0 and config.evaluation_interval and step % config.evaluation_interval == 0) or last_step:

        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=dtype):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    model.train()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):

        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # Forward pass
        with torch.autocast(device_type=device_type, dtype=dtype):
            logits, loss = model(x, y)

        # Backward pass
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        scaler.scale(loss).backward()

    # Clip the gradient to norm 1
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Get lr for this iteration
    lr = lr_scheduler.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Do one step and update weigths
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt

    print(f"step {step:5d} | loss: {loss_accum.item():9.6f} | lr {lr:.4e} | norm: {norm:7.4f} | dt: {dt:5.2f}s | tok/sec: {tokens_per_sec:.2f}")


if config.enable_wandb_logging:
    # Finish the W&B run
    wandb.finish()