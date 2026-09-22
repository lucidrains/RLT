import pytest
import torch
from RLT import RLT, Scale, slice_recurrent_lengths

param = pytest.mark.parametrize

def exists(v):
    return v is not None

def skip_if_flex_attn_unsupported(use_flex_attn):
    if use_flex_attn and not torch.cuda.is_available():
        pytest.skip('flex attention only supports backward on cuda')

@param('tie_embedding', (False, True))
@param('glu_cross', (False, True))
@param('dec_depth_scale_residual', (False, True))
@param('custom_recurrent_lengths', (False, True))
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
    tie_embedding,
    glu_cross,
    dec_depth_scale_residual,
    custom_recurrent_lengths,
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
        next_lat_loss = next_latent_prediction,
        glu_cross = glu_cross,
        dec_depth_scale_residual = dec_depth_scale_residual,
        tie_embedding = tie_embedding
    )

    if exists(num_tokens):
        if tie_embedding:
            assert model.to_logits[1].weight is model.token_emb.weight
        else:
            assert model.to_logits[1].weight is not model.token_emb.weight

    if exists(num_tokens):
        tokens = torch.randint(0, num_tokens, (2, seq_len))
        expected_shape = (2, seq_len, num_tokens)
    else:
        tokens = torch.randn(2, seq_len, 64)
        expected_shape = (2, seq_len, 64)

    recurrent_lengths = None

    if custom_recurrent_lengths:
        eff_seq_len = seq_len - 1 if (return_loss and not next_latent_prediction and exists(num_tokens)) else seq_len
        recurrent_lengths = slice_recurrent_lengths((1, 2, 5, 2, 1, 3, 2, 4, 1, 2), eff_seq_len)

    out = model(tokens, return_loss = return_loss, recurrent_lengths = recurrent_lengths)

    if not return_loss:
        out, _ = out
        assert out.shape == expected_shape

    out.sum().backward()

    if exists(num_tokens):
        sampled = model.generate(tokens, max_len = seq_len + 5)
        assert sampled.shape == (2, 5)

@param('recurrent_lengths', (
    (1, 2, 5, 2, 1),
    (2, 2, 2, 2, 2, 1),
    (11,),
    (1,) * 11,
    (3, 8),
))
@param('attn_residual', (False, True))
@param('num_tokens', (None, 256))
def test_custom_recurrent_lengths(
    recurrent_lengths,
    attn_residual,
    num_tokens
):
    model = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        num_tokens = num_tokens,
        attn_residual = attn_residual,
        dec_sliding_window_size = 4
    )

    seq_len = sum(recurrent_lengths)

    if exists(num_tokens):
        tokens = torch.randint(0, num_tokens, (2, seq_len))
        expected_shape = (2, seq_len, num_tokens)
    else:
        tokens = torch.randn(2, seq_len, 64)
        expected_shape = (2, seq_len, 64)

    out, _ = model(tokens, recurrent_lengths = recurrent_lengths)
    assert out.shape == expected_shape

    out.sum().backward()

    # validate sequential vs parallel with custom recurrent lengths

    model.eval()

    parallel_out, _ = model(tokens, recurrent_lengths = recurrent_lengths)

    memories = None
    sequential_outs = []
    curr = 0

    for block_len in recurrent_lengths:
        chunk = tokens[:, curr : curr + block_len]
        step_out, memories = model(chunk, memories = memories, recurrent_lengths = (block_len,))
        sequential_outs.append(step_out)
        curr += block_len

    sequential_out = torch.cat(sequential_outs, dim = 1)
    assert torch.allclose(parallel_out, sequential_out, atol = 1e-5)

    # validate that invalid recurrent lengths trigger lucidrains assert

    with pytest.raises(AssertionError):
        model(tokens, recurrent_lengths = (*recurrent_lengths, 1))

    # validate slice_recurrent_lengths and generate with custom recurrent lengths

    prompt_len = min(3, seq_len)
    prompt_lengths = slice_recurrent_lengths(recurrent_lengths, prompt_len)
    assert sum(prompt_lengths) == prompt_len

    if exists(num_tokens):
        prompt = tokens[:, :prompt_len]
        sampled = model.generate(prompt, recurrent_lengths = recurrent_lengths)
        assert sampled.shape == (2, seq_len - prompt_len)

        sampled_block_size = model.generate(prompt, max_len = seq_len, recurrent_lengths = 2)
        assert sampled_block_size.shape == (2, seq_len - prompt_len)

@param('prompt_len', (1, 2, 3, 5))
@param('block_size', (1, 2, 3))
def test_custom_recurrent_lengths_equivalent_to_fixed_block_size(
    block_size,
    prompt_len
):
    seq_len = 12

    model = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        num_tokens = 256,
        dec_sliding_window_size = 4,
        recurrent_block_size = block_size
    )

    tokens = torch.randint(0, 256, (2, seq_len))
    recurrent_lengths = (block_size,) * (seq_len // block_size)

    model.eval()

    fixed_out, _ = model(tokens)
    custom_out, _ = model(tokens, recurrent_lengths = recurrent_lengths)

    assert torch.allclose(fixed_out, custom_out, atol = 1e-5)

    prompt = tokens[:, :prompt_len]

    fixed_sampled = model.generate(prompt, max_len = seq_len, temperature = 0.)
    custom_sampled = model.generate(prompt, recurrent_lengths = recurrent_lengths, temperature = 0.)

    assert torch.equal(fixed_sampled, custom_sampled)

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

def test_shared_weights_equal_depth():
    model = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        num_tokens = 256,
        dec_sliding_window_size = 4,
        shared_weights = True
    )

    assert model.shared_weights

    # verify that self_attn and ff are shared across encoder and decoder layers

    for enc_layer, dec_layer in zip(model.encoder.layers, model.decoder.layers):
        enc_self_attn_norm, enc_self_attn, _, _, enc_ff_norm, enc_ff, _ = enc_layer
        dec_self_attn_norm, dec_self_attn, _, _, dec_ff_norm, dec_ff, _ = dec_layer

        assert enc_self_attn is dec_self_attn
        assert enc_ff is dec_ff

        # verify stage-specific normalizations remain separate

        assert enc_self_attn_norm is not dec_self_attn_norm
        assert enc_ff_norm is not dec_ff_norm

    # forward and backward

    tokens = torch.randint(0, 256, (2, 16))
    loss = model(tokens, return_loss = True)
    loss.backward()

    for enc_layer, dec_layer in zip(model.encoder.layers, model.decoder.layers):
        enc_self_attn_norm, enc_self_attn, _, _, enc_ff_norm, enc_ff, _ = enc_layer
        dec_self_attn_norm, dec_self_attn, _, _, dec_ff_norm, dec_ff, _ = dec_layer

        dec_self_attn = dec_self_attn.fn if isinstance(dec_self_attn, Scale) else dec_self_attn
        dec_ff = dec_ff.fn if isinstance(dec_ff, Scale) else dec_ff

        assert dec_self_attn.to_queries.weight.grad is not None
        assert dec_ff[0].weight.grad is not None
        assert enc_self_attn_norm.weight.grad is not None
        assert dec_self_attn_norm.weight.grad is not None
        assert enc_ff_norm.weight.grad is not None
        assert dec_ff_norm.weight.grad is not None

    # generate

    prompt = torch.randint(0, 256, (2, 4))
    sampled = model.generate(prompt, max_len = 16)
    assert sampled.shape == (2, 12)

    # asserting unequal depths when sharing weights

    with pytest.raises(AssertionError):
        RLT(dim = 64, enc_depth = 3, dec_depth = 2, shared_weights = True)

def test_stack_trans_layer():
    from stack_attention import StackTransLayer

    model = RLT(
        dim = 64,
        enc_depth = 2,
        dec_depth = 2,
        num_tokens = 256,
        recurrent_state_module = StackTransLayer(dim = 64)
    )

    tokens = torch.randint(0, 256, (2, 8))
    loss = model(tokens, return_loss = True)
    loss.backward()

    assert exists(model.recurrent_state_module.to_action_logits.weight.grad)
