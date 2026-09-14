import pytest
import torch
from RLT import RLT

param = pytest.mark.parametrize

def exists(v):
    return v is not None

@param('use_flex_attn', (False, True))
@param('seq_len', (2, 16))
@param('window_size', (1, 4))
@param('num_tokens, return_loss', (
    (None, False),
    (256, False),
    (256, True)
))
def test_rlt(
    use_flex_attn,
    seq_len,
    window_size,
    num_tokens,
    return_loss
):
    model = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        num_tokens = num_tokens,
        dec_sliding_window_size = window_size,
        use_flex_attn = use_flex_attn,
        tbptt_step_size = 2
    )

    if exists(num_tokens):
        tokens = torch.randint(0, num_tokens, (2, seq_len))
        expected_shape = (2, seq_len, num_tokens)
    else:
        tokens = torch.randn(2, seq_len, 64)
        expected_shape = (2, seq_len, 64)

    out = model(tokens, return_loss = return_loss)

    if not return_loss:
        out, _ = out
        assert out.shape == expected_shape

    out.sum().backward()

    if exists(num_tokens):
        sampled = model.generate(tokens, max_len = seq_len + 5)
        assert sampled.shape == (2, 5)

@param('use_flex_attn', (False, True))
@param('num_tokens', (None, 256))
def test_sequential_vs_parallel(use_flex_attn, num_tokens):
    model = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        num_tokens = num_tokens,
        dec_sliding_window_size = 4,
        use_flex_attn = use_flex_attn
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

@param('num_tokens', (None, 256))
def test_flex_vs_manual(num_tokens):
    torch.manual_seed(42)
    model = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        num_tokens = num_tokens,
        dec_sliding_window_size = 4,
        use_flex_attn = False
    )

    model_flex = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        num_tokens = num_tokens,
        dec_sliding_window_size = 4,
        use_flex_attn = True
    )

    model_flex.load_state_dict(model.state_dict())

    model.eval()
    model_flex.eval()

    if exists(num_tokens):
        tokens = torch.randint(0, num_tokens, (2, 16))
    else:
        tokens = torch.randn(2, 16, 64)

    parallel_out, _ = model(tokens)
    parallel_out_flex, _ = model_flex(tokens)

    assert torch.allclose(parallel_out, parallel_out_flex, atol = 1e-5)
