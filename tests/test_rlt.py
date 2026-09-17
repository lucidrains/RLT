import pytest
import torch
from RLT import RLT

param = pytest.mark.parametrize

def exists(v):
    return v is not None

def skip_if_flex_attn_unsupported(use_flex_attn):
    if use_flex_attn and not torch.cuda.is_available():
        pytest.skip('flex attention only supports backward on cuda')

@param('next_latent_prediction', (False, True))
@param('attn_residual', (False, True))
@param('use_flex_attn', (False, True))
@param('seq_len', (2, 16))
@param('window_size', (1, 4))
@param('num_tokens, return_loss', (
    (None, False),
    (256, False),
    (256, True)
))
@param('kv_heads, cross_attn_kv_heads', (
    (None, None),
    (2, None),
    (None, 2)
))
def test_rlt(
    next_latent_prediction,
    attn_residual,
    use_flex_attn,
    seq_len,
    window_size,
    num_tokens,
    return_loss,
    kv_heads,
    cross_attn_kv_heads
):
    skip_if_flex_attn_unsupported(use_flex_attn)

    model = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        num_tokens = num_tokens,
        kv_heads = kv_heads,
        cross_attn_kv_heads = cross_attn_kv_heads,
        dec_sliding_window_size = window_size,
        use_flex_attn = use_flex_attn,
        tbptt_step_size = 2,
        attn_residual = attn_residual,
        next_lat_loss = next_latent_prediction
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

@param('attn_residual', (False, True))
@param('use_flex_attn', (False, True))
@param('recurrent_block_size', (1, 4))
@param('num_tokens', (None, 256))
@param('kv_heads', (None, 2))
@param('num_kv_layers, layer_to_layer_mapping', (
    (None, None),
    (1, None),
    (2, None),
    (None, (0, 0)),
    (3, (0, 2))
))
def test_sequential_vs_parallel(
    attn_residual,
    use_flex_attn,
    recurrent_block_size,
    num_tokens,
    kv_heads,
    num_kv_layers,
    layer_to_layer_mapping
):
    skip_if_flex_attn_unsupported(use_flex_attn)

    model = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        num_tokens = num_tokens,
        kv_heads = kv_heads,
        num_kv_layers = num_kv_layers,
        layer_to_layer_mapping = layer_to_layer_mapping,
        dec_sliding_window_size = 4,
        recurrent_block_size = recurrent_block_size,
        use_flex_attn = use_flex_attn,
        attn_residual = attn_residual
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

@param('attn_residual', (False, True))
@param('num_tokens', (None, 256))
@param('kv_heads', (None, 2))
def test_flex_vs_manual(attn_residual, num_tokens, kv_heads):
    skip_if_flex_attn_unsupported(True)

    torch.manual_seed(42)
    model = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        num_tokens = num_tokens,
        kv_heads = kv_heads,
        dec_sliding_window_size = 4,
        use_flex_attn = False,
        attn_residual = attn_residual
    )

    model_flex = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        num_tokens = num_tokens,
        kv_heads = kv_heads,
        dec_sliding_window_size = 4,
        use_flex_attn = True,
        attn_residual = attn_residual
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
