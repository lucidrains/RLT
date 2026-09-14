import pytest
import torch
from RLT import RLT

param = pytest.mark.parametrize

@param('seq_len', (1, 16))
@param('window_size', (1, 4))
@param('batch_size', (1, 2))
def test_rlt(
    seq_len,
    window_size,
    batch_size
):
    model = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        dec_sliding_window_size = window_size
    )

    tokens = torch.randn(batch_size, seq_len, 64)
    out = model(tokens)

    assert out.shape == tokens.shape

    out.sum().backward()
