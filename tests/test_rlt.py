import pytest
import torch
from RLT import RLT

param = pytest.mark.parametrize

def exists(v):
    return v is not None

@param('seq_len', (1, 16))
@param('window_size', (1, 4))
@param('num_tokens', (None, 256))
def test_rlt(
    seq_len,
    window_size,
    num_tokens
):
    model = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        num_tokens = num_tokens,
        dec_sliding_window_size = window_size
    )

    if exists(num_tokens):
        tokens = torch.randint(0, num_tokens, (2, seq_len))
        expected_shape = (2, seq_len, num_tokens)
    else:
        tokens = torch.randn(2, seq_len, 64)
        expected_shape = (2, seq_len, 64)

    out, _ = model(tokens)

    assert out.shape == expected_shape

    out.sum().backward()

@param('num_tokens', (None, 256))
def test_sequential_vs_parallel(num_tokens):
    model = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        num_tokens = num_tokens,
        dec_sliding_window_size = 4
    )

    model.eval()

    seq_len = 16

    if exists(num_tokens):
        tokens = torch.randint(0, num_tokens, (2, seq_len))
    else:
        tokens = torch.randn(2, seq_len, 64)

    parallel_out, _ = model(tokens)

    memories = None
    sequential_outs = []

    for token in tokens.unbind(dim = 1):
        step_out, memories = model(token[:, None], memories = memories)
        sequential_outs.append(step_out)

    sequential_out = torch.cat(sequential_outs, dim = 1)

    assert torch.allclose(parallel_out, sequential_out, atol = 1e-5)
