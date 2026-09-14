# /// script
# dependencies = [
#   "rlt-pytorch",
#   "accelerate",
#   "fire",
#   "numpy",
#   "tqdm",
# ]
# ///

from __future__ import annotations

import gzip
import random
import tqdm
import fire
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

from RLT import RLT

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# dataset

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

# main training function

def train(
    num_batches: int = 10_000,
    batch_size: int = 4,
    grad_accum_every: int = 2,
    learning_rate: float = 1e-3,
    validate_every: int = 100,
    generate_every: int = 100,
    prime_length: int = 32,
    generate_length: int = 64,
    seq_len: int = 64,
    dim: int = 256,
    enc_depth: int = 6,
    dec_depth: int = 2,
    dec_sliding_window_size: int = 8,
    recurrent_transition_alpha: float = 1.,
    filter_thres: float = 0.9,
    temperature: float = 1.,
    cpu: bool = False,
    data_path: str = "./data/enwik8.gz"
):
    accelerator = Accelerator(cpu = cpu)

    # prepare enwik8 data

    with gzip.open(data_path) as file:
        data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset = TextSamplerDataset(data_val, seq_len)
    train_loader = DataLoader(train_dataset, batch_size = batch_size)
    val_loader = DataLoader(val_dataset, batch_size = batch_size)

    # model & optimizer

    model = RLT(
        num_tokens = 256,
        dim = dim,
        enc_depth = enc_depth,
        dec_depth = dec_depth,
        dec_sliding_window_size = dec_sliding_window_size,
        recurrent_transition_alpha = recurrent_transition_alpha
    )

    optim = Adam(model.parameters(), lr = learning_rate)

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    # training

    pbar = tqdm.tqdm(range(1, num_batches + 1), mininterval = 2.0, desc = "training")

    for step in pbar:
        model.train()
        total_loss = 0.

        for _ in range(grad_accum_every):
            data = next(train_loader)

            loss = model(data, return_loss = True)

            accelerator.backward(loss / grad_accum_every)
            total_loss += loss.item()

        accelerator.clip_grad_norm_(model.parameters(), 0.5)

        optim.step()
        optim.zero_grad()

        avg_loss = total_loss / grad_accum_every
        pbar.set_postfix(loss = f"{avg_loss:.3f}")

        if divisible_by(step, validate_every):
            model.eval()
            with torch.no_grad():
                valid_data = next(val_loader)
                val_loss = model(valid_data, return_loss = True)
                accelerator.print(f"\n[Step {step}] validation loss: {val_loss.item():.3f}")

        if divisible_by(step, generate_every) or step == num_batches:
            model.eval()
            unwrapped_model = accelerator.unwrap_model(model)

            inp = random.choice(val_dataset)[:prime_length]
            inp = inp.to(accelerator.device)

            prime = decode_tokens(inp)
            accelerator.print(f"\n--- [Step {step}] GENERATION ---")
            accelerator.print(f"PROMPT: {prime}")

            prompt = inp[None, ...]

            sampled = unwrapped_model.generate(
                prompt,
                max_len = generate_length,
                temperature = temperature,
                filter_kwargs = dict(thres = filter_thres)
            )

            decoded_output = decode_tokens(sampled[0])
            accelerator.print(f"OUTPUT: {decoded_output}\n")

    accelerator.end_training()
    accelerator.print("Training complete!")

if __name__ == "__main__":
    fire.Fire(train)
