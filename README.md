<img src="./rlt.png" width="400"></img>

## RLT (Recurrent Looped Transformer)

Unofficial implementation of the [Recurrent Looped Transformer](https://yifanzhang-pro.github.io/recurrent-looped-tranformer/) proposed by Yifan Zhang of Princeton.

## Appreciation

- [Pranoy](https://github.com/pranoyr) for the PR on weight tying / sharing between encoder and decoder!

## Install

```bash
$ pip install rlt-pytorch
```

## Usage

```python
import torch
from RLT import RLT

model = RLT(
    num_tokens = 256,
    dim = 512,
    enc_depth = 4,
    dec_depth = 4,
    dec_sliding_window_size = 16,
    tbptt_step_size = 16 # optional truncated bptt
)

tokens = torch.randint(0, 256, (2, 1024))

# forward for loss

loss = model(tokens, return_loss = True)
loss.backward()

# generate

prompt = torch.randint(0, 256, (2, 32))

sampled = model.generate(prompt, max_len = 128) # (2, 96)
```

To turn on [Next-Latent Prediction](https://arxiv.org/abs/2511.05963) (Teoh et al.):

```python
model = RLT(
    num_tokens = 256,
    dim = 512,
    enc_depth = 4,
    dec_depth = 4,
    next_lat_loss = True
)

loss = model(tokens, return_loss = True)
loss.backward()
```

The recurrent block size can also vary across the sequence by passing `recurrent_lengths` - a sequence of block lengths that must sum to the sequence length

```python
block_lengths = (1, 2, 5, 2, 1, 3)

tokens = torch.randint(0, 256, (2, sum(block_lengths)))

loss = model(tokens, return_loss = True, recurrent_lengths = block_lengths)
loss.backward()

# during generation, the recurrent state only advances at the block boundaries

prompt = torch.randint(0, 256, (2, 2))

sampled = model.generate(prompt, recurrent_lengths = block_lengths)
```

## Test

Train on enwik8

```bash
$ uv run train_enwik8.py
```

## Citations

```bibtex
@techreport{zhang2026recurrentlooped,
    title  = {Recurrent Looped Transformer},
    author = {Zhang, Yifan},
    year   = {2026},
    month  = {Sep},
    url    = {https://github.com/yifanzhang-pro/recurrent-looped-tranformer}
}
```

```bibtex
@misc{kimiteam2026attentionresiduals,
    title   = {Attention Residuals},
    author  = {Kimi Team and Guangyu Chen and Yu Zhang and Jianlin Su and Weixin Xu and Siyuan Pan and Yaoyu Wang and Yucheng Wang and Guanduo Chen and Bohong Yin and Yutian Chen and Junjie Yan and Ming Wei and Y. Zhang and Fanqing Meng and Chao Hong and Xiaotong Xie and Shaowei Liu and Enzhe Lu and Yunpeng Tai and Yanru Chen and Xin Men and Haiqing Guo and Y. Charles and Haoyu Lu and Lin Sui and Jinguo Zhu and Zaida Zhou and Weiran He and Weixiao Huang and Xinran Xu and Yuzhi Wang and Guokun Lai and Yulun Du and Yuxin Wu and Zhilin Yang and Xinyu Zhou},
    year    = {2026},
    eprint  = {2603.15031},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/2603.15031},
}
```

```bibtex
@misc{teoh2025nextlatentpredictiontransformerslearn,
    title     = {Next-Latent Prediction Transformers Learn Compact World Models},
    author    = {Jayden Teoh and Manan Tomar and Kwangjun Ahn and Edward S. Hu and Tim Pearce and Pratyusha Sharma and Akshay Krishnamurthy and Riashat Islam and Alex Lamb and John Langford},
    year      = {2025},
    eprint    = {2511.05963},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url       = {https://arxiv.org/abs/2511.05963}
}
```

```bibtex
@misc{wang2026fullbandwidthtransformer,
    title         = {Full-bandwidth transformer},
    author        = {Xi Wang and Ziyang Cai and Zheng Zhan and Harry Dong and Ying Fan and Gustavo de Rosa and Tim Pearce and John Langford},
    year          = {2026},
    eprint        = {2608.08888},
    archivePrefix = {arXiv},
    primaryClass  = {cs.LG},
    url           = {https://arxiv.org/abs/2608.08888}
}
```
