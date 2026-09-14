import pytest
import torch
from RLT import RLT, Transformer

def test_rlt():
    model = RLT(
        dim = 512,
        enc_depth = 2,
        dec_depth = 2
    )

    tokens = torch.randn(2, 16, 512)
    out = model(tokens)

    assert out.shape == tokens.shape

    out.sum().backward()
