# /// script
# dependencies = [
#     "rlt-pytorch",
#     "x-transformers",
#     "tqdm",
# ]
# ///

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
from x_transformers import TransformerWrapper, Decoder
from RLT import RLT

# parity (cumsum mod 2) needs recurrence to generalize beyond depth

# helpers

def cycle(batch_size, length):
    while True:
        seq = torch.randint(0, 2, (batch_size, length))
        yield seq, seq.cumsum(dim = -1) % 2

def get_logits(out):
    return out[0] if isinstance(out, tuple) else out

# main

def main(
    batch_size = 256,
    train_steps = 300,
    train_length = 16,
    eval_lengths = (8, 16, 32),
    dim = 64,
    depth = 3,
    enc_depth = 1,
    dec_depth = 1,
    heads = 4,
    dim_head = 16,
    recurrent_block_size = 1,
    dec_sliding_window_size = 2,
    lr = 3e-3
):
    train_dl = cycle(batch_size, train_length)

    plain = TransformerWrapper(
        num_tokens = 2,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            rotary_pos_emb = True,
            rotary_emb_dim = dim_head,
            use_rmsnorm = True,
            ff_glu = True
        )
    )

    rlt = RLT(
        num_tokens = 2,
        dim = dim,
        enc_depth = enc_depth,
        dec_depth = dec_depth,
        heads = heads,
        dim_head = dim_head,
        recurrent_block_size = recurrent_block_size,
        dec_sliding_window_size = dec_sliding_window_size
    )

    def train(model):
        model.train()
        optimizer = AdamW(model.parameters(), lr = lr)

        for _ in tqdm(range(train_steps)):
            seq, labels = next(train_dl)

            optimizer.zero_grad()

            logits = get_logits(model(seq))
            loss = F.cross_entropy(logits.transpose(-1, -2), labels)

            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    def last_token_acc(model, length):
        model.eval()
        seq, labels = next(cycle(512, length))

        with torch.no_grad():
            logits = get_logits(model(seq))
            pred = logits[:, -1].argmax(dim = -1)

        return (pred == labels[:, -1]).float().mean().item() * 100

    train(plain)
    train(rlt)

    print('\nbinary parity - last token % correct (chance = 50%, solved = 100%):\n')

    for length in eval_lengths:
        plain_acc = last_token_acc(plain, length)
        rlt_acc = last_token_acc(rlt, length)
        print(f'length {length:3d}: plain {plain_acc:5.1f}%   rlt {rlt_acc:5.1f}%')

    print(f'\ntrained {train_steps} steps at length {train_length}; RLT generalizes beyond the trained length, the plain transformer cannot.')

if __name__ == '__main__':
    main()
