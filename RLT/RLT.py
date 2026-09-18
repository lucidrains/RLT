from __future__ import annotations

from collections import namedtuple
from functools import partial
from itertools import accumulate
from math import ceil
from typing import Callable, Sequence

import torch
from torch import nn, cat, tensor, Tensor, is_tensor
from torch.nn import Module, ModuleList, Linear, Sequential, RMSNorm, Parameter
import torch.nn.functional as F

from torch.func import functional_call

from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

from x_mlps_pytorch import MLP

from torch_einops_utils import masked_mean, maybe_return, pack_with_inverse, repeat_interleave_to_match, temp_eval, tree_map_detach, tree_map_tensor

# types

RLTMemories = namedtuple('RLTMemories', ['encoder_memories', 'decoder_memories'])
EncoderMemories = namedtuple('EncoderMemories', ['memories', 'keys_values'])
DecoderMemories = namedtuple('DecoderMemories', ['state', 'memories'])
TransformerMemories = namedtuple('TransformerMemories', ['step', 'memories'])
Losses = namedtuple('Losses', ['cross_entropy', 'next_latent', 'kl_div'])

# constants

LinearNoBias = partial(Linear, bias = False)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t, *args, **kwargs):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def to_tuple(t):
    if is_tensor(t):
        return tuple(t.tolist())
    elif isinstance(t, list):
        return tuple(t)

    return t

def slice_recurrent_lengths(recurrent_lengths, length: int, offset: int = 0):
    recurrent_lengths = to_tuple(recurrent_lengths)

    if not recurrent_lengths or length <= 0:
        return () if exists(recurrent_lengths) else None

    if isinstance(recurrent_lengths, int):
        first = min((-offset) % recurrent_lengths or recurrent_lengths, length)
        num_full, rem = divmod(length - first, recurrent_lengths)
        return (first,) + (recurrent_lengths,) * num_full + ((rem,) if rem else ())

    first, *rest = recurrent_lengths

    if offset >= first:
        return slice_recurrent_lengths(rest, length, offset - first)

    first_len = min(first - offset, length)
    return (first_len, *slice_recurrent_lengths(rest, length - first_len))

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, eps = 1e-10):
    if temperature == 0.:
        return t.argmax(dim = dim)

    return ((t / max(temperature, eps)) + gumbel_noise(t)).argmax(dim = dim)

# topk

def top_k(logits, frac_num_tokens = 0.1, k: int | None = None, thres: float | None = None):
    num_tokens = logits.shape[-1]

    if exists(thres):
        frac_num_tokens = 1. - thres

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(max(k, 1), num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

# maybe flex attention

try:
    from torch.nn.attention.flex_attention import flex_attention as pt_flex_attention, create_block_mask

    if torch.cuda.is_available():
        pt_flex_attention = torch.compile(pt_flex_attention)

except ImportError:
    pt_flex_attention = None
    create_block_mask = None

def flex_attention(
    q, k, v,
    causal = False,
    sliding_window_size = None,
    scale = None
):
    assert exists(pt_flex_attention), 'flex_attention requires torch >= 2.5.0'

    rows, cols = q.shape[-2], k.shape[-2]
    single_token = rows == 1

    block_mask = None

    if (causal or exists(sliding_window_size)) and not single_token:
        prefix_len = cols - rows

        def mask_mod(b, h, q_idx, kv_idx):
            mask = True
            if causal:
                mask = mask & (q_idx + prefix_len >= kv_idx)
            if exists(sliding_window_size):
                mask = mask & (q_idx + prefix_len - kv_idx < sliding_window_size)
            return mask

        block_mask = create_block_mask(mask_mod, B = None, H = None, Q_LEN = rows, KV_LEN = cols, device = q.device)

    return pt_flex_attention(q, k, v, block_mask = block_mask, scale = scale)

# lora linear

class LoRALinear(Module):
    def __new__(
        cls,
        dim_in,
        dim_out = None,
        rank = None
    ):
        dim_out = default(dim_out, dim_in)

        if not exists(rank):
            return LinearNoBias(dim_in, dim_out)

        return super().__new__(cls)

    def __init__(
        self,
        dim_in,
        dim_out = None,
        rank = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim_in)
        self.down = LinearNoBias(dim_in, rank)
        self.up = LinearNoBias(rank, dim_out)

    def forward(self, x):
        return self.up(self.down(x))

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        kv_heads: int | None = None,
        causal = True,
        cross_attend_key_values = False,
        rotary_embed: RotaryEmbedding | None = None,
        sliding_window_size: int | None = None,
        use_flex_attn = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        self.causal = causal

        kv_heads = default(kv_heads, heads)
        assert divisible_by(heads, kv_heads), f'heads ({heads}) must be divisible by kv_heads ({kv_heads})'

        dim_kv_inner = dim_head * kv_heads

        self.to_queries = LinearNoBias(dim, dim_inner)
        self.to_key_values = LinearNoBias(dim, dim_kv_inner * 2) if not cross_attend_key_values else None

        self.split_queries = Rearrange('b n (h d) -> b h n d', h = heads)
        self.split_key_values = Rearrange('b n (h d) -> b h n d', h = kv_heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = LinearNoBias(dim_inner, dim)

        self.rotary_embed = rotary_embed
        self.sliding_window_size = sliding_window_size

        assert not (use_flex_attn and not exists(pt_flex_attention)), 'flex attention is only available on torch 2.5.0 onwards'

        self.use_flex_attn = use_flex_attn

    def forward(
        self,
        tokens,
        keys_values = None,
        memories = None,
        offset = 0,
        return_memories = False,
        sliding_window_size = None
    ):
        sliding_window_size = default(sliding_window_size, self.sliding_window_size)

        q = self.to_queries(tokens)

        # keys and values can be received from the encoder

        if not exists(keys_values):
            assert exists(self.to_key_values), 'keys_values must be provided for cross attention'
            k, v = self.to_key_values(tokens).chunk(2, dim = -1)
            k, v = (self.split_key_values(t) for t in (k, v))
        else:
            k, v = keys_values

        q = self.split_queries(q)

        # rotary embedding

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q, offset = offset)
            k = self.rotary_embed.rotate_queries_or_keys(k, offset = offset)

        if exists(memories):
            mk, mv = memories
            k = cat((mk, k), dim = -2)
            v = cat((mv, v), dim = -2)

        return_kv = (k, v)

        # grouped query attention - repeat key / values to match the query heads, if needed

        k = repeat_interleave_to_match(k, q, dim = -3)
        v = repeat_interleave_to_match(v, q, dim = -3)

        rows, cols = q.shape[-2], k.shape[-2]
        single_token = rows == 1

        if self.use_flex_attn:
            out = flex_attention(
                q, k, v,
                causal = self.causal,
                sliding_window_size = sliding_window_size,
                scale = self.scale
            )
        else:
            sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

            mask = None
            prefix_len = cols - rows

            if self.causal and not single_token:
                causal_mask = torch.ones((rows, cols), dtype = torch.bool, device = sim.device).triu(prefix_len + 1)
                mask = causal_mask

            if exists(sliding_window_size) and not single_token:
                window_mask = torch.ones((rows, cols), dtype = torch.bool, device = sim.device).tril(prefix_len - sliding_window_size)
                mask = window_mask if mask is None else (mask | window_mask)

            if exists(mask):
                sim = sim.masked_fill(mask, max_neg_value(sim))

            attn = sim.softmax(dim = -1)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)
        out = self.to_out(out)

        if not return_memories:
            return out

        return out, return_kv

# feedforward

class GEGLU(Module):
    # Shazeer et al.

    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return F.gelu(gates) * x

def Feedforward(
    dim,
    expansion_factor = 4.
):
    dim_inner = int(dim * expansion_factor * 2 / 3)

    return Sequential(
        Linear(dim, dim_inner * 2),
        GEGLU(),
        Linear(dim_inner, dim)
    )

# attention residual
# Guangyu (Nathan) Chen et al. with Kimi team https://arxiv.org/abs/2603.15031

class AttentionResidual(Module):
    def __init__(
        self,
        dim,
        *,
        query_key_rank: int | None = None
    ):
        super().__init__()
        self.scale = dim ** -0.5

        self.to_queries = LoRALinear(dim, rank = query_key_rank)
        self.to_keys = LoRALinear(dim, rank = query_key_rank)

        self.query_rmsnorm = RMSNorm(dim)
        self.key_rmsnorm = RMSNorm(dim)

    def forward(
        self,
        block_outputs: list[Tensor] | Tensor
    ):
        block_outputs = list(block_outputs)
        curr_tokens = block_outputs[-1]

        past_layers = rearrange(block_outputs, 'l b n d -> b n l d')

        queries = self.to_queries(curr_tokens)
        keys = self.to_keys(past_layers)

        queries = self.query_rmsnorm(queries)
        keys = self.key_rmsnorm(keys)

        sim = einsum(queries, keys, 'b n d, b n l d -> b n l') * self.scale

        attn = sim.softmax(dim = -1)

        return einsum(attn, past_layers, 'b n l, b n l d -> b n d')

# next-latent prediction
# Jayden Teoh et al. https://arxiv.org/abs/2511.05963

class NextLatentPrediction(Module):
    def __init__(
        self,
        dim,
        depth = 3,
        num_rollouts = 1
    ):
        super().__init__()
        self.num_rollouts = num_rollouts

        self.norm = RMSNorm(dim * 2)

        dims = (dim * 2, *(dim,) * (depth - 1), dim)

        self.mlp = MLP(*dims, activation = nn.GELU(), bias = False)

        # zero init the last linear layer so dynamics starts as identity

        nn.init.zeros_(self.mlp.layers[-1].weight)

    def rollout_loss(
        self,
        hiddens,
        token_embeds,
        teacher_logits,
        to_logits,
        labels,
        kl_loss_weight = 1.
    ):
        num_rollouts = self.num_rollouts
        assert hiddens.shape[1] > num_rollouts, f'sequence length ({hiddens.shape[1]}) must be greater than num_rollouts ({num_rollouts})'

        total_latent_loss = 0.
        total_kl_loss = 0.

        pred_h, target_h, next_tokens = hiddens, hiddens, token_embeds
        mask = labels != -1

        has_kl = kl_loss_weight > 0.
        lm_head_params = {k: v.detach() for k, v in to_logits.named_parameters()} if has_kl else None

        for i in range(num_rollouts):
            if i > 0:
                mask = mask[:, 1:]

            pred_h = pred_h[:, :-1]
            next_tokens = next_tokens[:, 1:]
            target_h = target_h[:, 1:]

            pred_h = self(pred_h, next_tokens)

            # next latent loss, stop-gradient on the target

            latent_loss = F.smooth_l1_loss(pred_h, target_h.detach(), reduction = 'none').mean(dim = -1)
            total_latent_loss = total_latent_loss + masked_mean(latent_loss, mask)

            # distill teacher next token logits through a frozen lm head

            if has_kl and teacher_logits.shape[1] > 1:
                teacher_logits = teacher_logits[:, 1:]

                student_logits = functional_call(to_logits, lm_head_params, pred_h[:, :-1])

                log_p = F.log_softmax(student_logits, dim = -1)
                log_q = F.log_softmax(teacher_logits.detach(), dim = -1)

                kl = F.kl_div(log_p, log_q, log_target = True, reduction = 'none').sum(dim = -1)
                total_kl_loss = total_kl_loss + masked_mean(kl, mask[:, 1:])

        return total_latent_loss / num_rollouts, total_kl_loss / num_rollouts

    def forward(self, current_states, next_token_embeds):
        x = cat((current_states, next_token_embeds), dim = -1)
        x = self.norm(x)
        return self.mlp(x) + current_states

# transformer

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        kv_heads: int | None = None,
        cross_attn_kv_heads: int | None = None,
        self_attn = True,
        cross_attn = False,
        self_attn_window_size = None,
        ff_expansion_factor = 4,
        rotary_embed = True,
        dim_rotary = None,
        use_flex_attn = False,
        attn_residual = False,
        attn_residual_query_key_rank: int | None = None
    ):
        super().__init__()
        assert not exists(self_attn_window_size) or self_attn_window_size >= 1
        assert not exists(dim_rotary) or dim_rotary <= dim_head

        self.self_attn_window_size = self_attn_window_size
        self.has_attn_residual = attn_residual

        # rotary embedding

        dim_rotary = default(dim_rotary, dim_head)
        self.rotary_embed = RotaryEmbedding(dim_rotary) if rotary_embed else None

        layers = ModuleList([])

        for _ in range(depth):
            self_attn_norm = RMSNorm(dim) if self_attn else None
            self_attn_module = Attention(dim = dim, dim_head = dim_head, heads = heads, kv_heads = kv_heads, rotary_embed = self.rotary_embed, sliding_window_size = self.self_attn_window_size, use_flex_attn = use_flex_attn) if self_attn else None

            cross_attn_norm = RMSNorm(dim) if cross_attn else None
            cross_attn_module = Attention(dim = dim, dim_head = dim_head, heads = heads, kv_heads = cross_attn_kv_heads, cross_attend_key_values = True, use_flex_attn = use_flex_attn) if cross_attn else None

            ff_norm = RMSNorm(dim)
            ff = Feedforward(dim = dim, expansion_factor = ff_expansion_factor)

            attn_res = AttentionResidual(dim, query_key_rank = attn_residual_query_key_rank) if attn_residual else None

            layers.append(ModuleList([self_attn_norm, self_attn_module, cross_attn_norm, cross_attn_module, ff_norm, ff, attn_res]))

        self.layers = layers

    @maybe_return('memories', 'hiddens')
    def forward(
        self,
        tokens,
        keys_values = None,
        memories = None,
        block_outputs: list[Tensor] | None = None,
        return_hiddens = False
    ):
        seq_len = tokens.shape[-2]

        step, memories = default(memories, (0, []))
        iter_memories = iter(memories)
        next_memories = []

        # keep the hidden states only if they are needed for the attention residual, or explicitly requested

        keep_hiddens = return_hiddens or self.has_attn_residual or exists(block_outputs)

        if keep_hiddens:
            block_outputs = [block_outputs] if is_tensor(block_outputs) else list(default(block_outputs, [tokens]))

        # keys and values can either be a single (keys, values) pair shared by all layers, or one pair per layer

        if exists(keys_values):
            if is_tensor(keys_values[0]):
                keys_values = (keys_values,) * len(self.layers)
            else:
                assert len(keys_values) == len(self.layers), f'expected {len(self.layers)} (keys, values) pairs, received {len(keys_values)}'

        iter_keys_values = iter(default(keys_values, ()))

        # layers

        for self_attn_norm, self_attn, cross_attn_norm, cross_attn, ff_norm, ff, attn_residual in self.layers:

            # self attention

            if exists(self_attn):
                self_attn_out, next_memory = self_attn(
                    self_attn_norm(tokens),
                    memories = next(iter_memories, None),
                    offset = step,
                    return_memories = True,
                    sliding_window_size = self.self_attn_window_size
                )
                tokens = self_attn_out + tokens

                next_memories.append(next_memory)

            # special cross attention from YOCO

            if exists(cross_attn):
                layer_keys_values = next(iter_keys_values, None)
                tokens = cross_attn(cross_attn_norm(tokens), keys_values = layer_keys_values) + tokens

            # feedforward

            tokens = ff(ff_norm(tokens)) + tokens

            # keep the hidden states for the attention residual

            if keep_hiddens:
                block_outputs.append(tokens)

            # attention residual

            if exists(attn_residual):
                tokens = attn_residual(block_outputs)

        # maybe take care of sliding window size - since always doing one token at a time, just do like inference where one slices off the earlier end

        if exists(self.self_attn_window_size):
            w = self.self_attn_window_size
            next_memories = tree_map_tensor(lambda t: t[..., -(w - 1):, :] if w > 1 else t[..., :0, :], next_memories)

        next_step = step + seq_len

        return tokens, TransformerMemories(next_step, next_memories), block_outputs

# the recurrent transition they propose

class RecurrentTransition(Module):
    def __init__(
        self,
        dim,
        alpha = 1.,
        learned_alpha = False
    ):
        super().__init__()
        self.learned_alpha = learned_alpha
        self.alpha = Parameter(tensor(alpha)) if learned_alpha else alpha

        self.norm = RMSNorm(dim)
        self.to_gates = Linear(dim * 2, dim)
        self.to_state = LinearNoBias(dim, dim)

    def forward(
        self,
        state,
        encoded
    ):
        α = F.softplus(self.alpha) if self.learned_alpha else self.alpha

        normed_state = self.norm(state)
        encoded, normed_state = torch.broadcast_tensors(encoded, normed_state)
        gates = self.to_gates(cat((encoded, normed_state), dim = -1)).sigmoid()
        state_out = self.to_state(normed_state)

        # section 2.5, equations (2.9) - (2.11)

        return encoded + α * gates * state_out

# main class

class RLT(Module):
    def __init__(
        self,
        dim,
        *,
        enc_depth,
        dec_depth,
        num_tokens = None,
        dim_head = 64,
        heads = 8,
        kv_heads: int | None = None,
        cross_attn_kv_heads: int | None = None,
        num_kv_layers: int | None = None,
        layer_to_layer_mapping: Sequence[int] | None = None,
        ff_expansion_factor = 4.,
        recurrent_transition_alpha = 1.,
        dec_sliding_window_size = 16,
        recurrent_block_size: int | Sequence[int] = 1,
        rotary_embed = True,
        dim_rotary = None,
        recurrent_transition: Module | None = None,
        use_flex_attn = False,
        tbptt_step_size: int | None = None,
        attn_residual = False,
        attn_residual_query_key_rank: int | None = None,
        attn_residual_cross_encoder = False,
        shared_weights = False,
        next_lat_loss = False,
        next_latent_loss_weight = 0.,
        next_latent_kl_loss_weight = 1.,
        next_latent_num_rollouts = 1,
        next_latent_dynamics_depth = 3
    ):
        super().__init__()
        has_num_tokens = exists(num_tokens)
        self.has_num_tokens = has_num_tokens
        self.recurrent_block_size = to_tuple(recurrent_block_size)

        self.token_emb = nn.Embedding(num_tokens, dim) if has_num_tokens else None

        # next latent prediction (Teoh et al. https://arxiv.org/abs/2511.05963)

        if next_lat_loss and next_latent_loss_weight == 0.:
            next_latent_loss_weight = 1.

        self.has_next_latent_loss = has_num_tokens and next_latent_loss_weight > 0.
        self.next_latent_loss_weight = next_latent_loss_weight
        self.next_latent_kl_loss_weight = next_latent_kl_loss_weight

        self.next_latent_prediction = NextLatentPrediction(
            dim = dim,
            depth = next_latent_dynamics_depth,
            num_rollouts = next_latent_num_rollouts
        ) if self.has_next_latent_loss else None

        # resolve the mapping from decoder layers to the encoded key / value layers

        if exists(layer_to_layer_mapping):
            layer_to_layer_mapping = tuple(layer_to_layer_mapping)
            num_kv_layers = default(num_kv_layers, max(layer_to_layer_mapping) + 1)

            assert len(layer_to_layer_mapping) == dec_depth, f'layer_to_layer_mapping must have length {dec_depth}'
            assert all(0 <= layer_index < num_kv_layers for layer_index in layer_to_layer_mapping), f'layer_to_layer_mapping indices must be in the range [0, {num_kv_layers})'
        else:
            num_kv_layers = default(num_kv_layers, 1)

            assert num_kv_layers in (1, dec_depth), 'layer_to_layer_mapping must be specified if num_kv_layers is neither 1 nor equal to dec_depth'

            layer_to_layer_mapping = tuple(range(dec_depth)) if num_kv_layers == dec_depth else (0,) * dec_depth

        self.layer_to_layer_mapping = layer_to_layer_mapping

        # grouped query attention - `kv_heads` applies to all attention, `cross_attn_kv_heads` overrides the cross attention

        cross_attn_kv_heads = default(cross_attn_kv_heads, default(kv_heads, heads))

        assert divisible_by(heads, cross_attn_kv_heads), f'heads ({heads}) must be divisible by cross_attn_kv_heads ({cross_attn_kv_heads})'

        dim_kv_inner = dim_head * cross_attn_kv_heads

        # whether the decoder attention residual should also attend to the encoder hiddens

        assert not attn_residual_cross_encoder or attn_residual, 'attn_residual must be enabled for the decoder attention residual to attend to encoder hiddens'

        self.attn_residual = attn_residual
        self.attn_residual_cross_encoder = attn_residual_cross_encoder

        # transformer settings

        transformer_kwargs = dict(
            heads = heads,
            kv_heads = kv_heads,
            dim_head = dim_head,
            ff_expansion_factor = ff_expansion_factor,
            rotary_embed = rotary_embed,
            dim_rotary = dim_rotary,
            use_flex_attn = use_flex_attn,
            attn_residual = attn_residual,
            attn_residual_query_key_rank = attn_residual_query_key_rank
        )

        assert not (shared_weights and enc_depth != dec_depth), f'enc_depth ({enc_depth}) must equal dec_depth ({dec_depth}) when sharing weights'

        self.shared_weights = shared_weights

        # encoder

        self.encoder = Transformer(dim, depth = enc_depth, **transformer_kwargs)

        # YOCO - Sun et al.

        self.to_encoded_key_values = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, num_kv_layers * dim_kv_inner * 2)
        )

        self.split_heads = Rearrange('b n (l h d) -> l b h n d', l = num_kv_layers, h = cross_attn_kv_heads)

        # recurrence related

        self.tbptt_step_size = tbptt_step_size

        self.initial_state = nn.Parameter(torch.randn(dim) * 1e-2)

        if not exists(recurrent_transition):
            recurrent_transition = RecurrentTransition(dim, alpha = recurrent_transition_alpha)

        self.combine_encoded_token_and_state = recurrent_transition

        # decoder

        self.decoder = Transformer(
            dim,
            depth = dec_depth,
            cross_attn = True,
            cross_attn_kv_heads = cross_attn_kv_heads,
            self_attn_window_size = dec_sliding_window_size,
            **transformer_kwargs
        )

        # maybe share weights between encoder and decoder
        # stage-specific normalizations remain separate in each Transformer, while self_attn and ff are shared

        if self.shared_weights:
            for enc_layer, dec_layer in zip(self.encoder.layers, self.decoder.layers):
                enc_layer[1] = dec_layer[1]
                enc_layer[5] = dec_layer[5]

        # to logits

        self.to_logits = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, num_tokens)
        ) if has_num_tokens else None

    @property
    def device(self):
        return next(self.parameters()).device

    # generate

    @temp_eval
    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        seq_len: int | None = None,
        max_len: int | None = None,
        temperature: float = 1.,
        filter_fn: Callable = top_k,
        filter_kwargs: dict = dict(frac_num_tokens = 0.1),
        recurrent_lengths: int | Sequence[int] | None = None
    ):
        assert self.has_num_tokens, '`num_tokens` must be passed to RLT to generate'

        recurrent_lengths = default(recurrent_lengths, self.recurrent_block_size)
        is_custom = not isinstance(recurrent_lengths, int)

        max_len = default(max_len, sum(to_tuple(recurrent_lengths)) if is_custom else seq_len)
        assert exists(max_len), 'max_len must be supplied'

        recurrent_lengths = slice_recurrent_lengths(recurrent_lengths, max_len)

        assert all(isinstance(l, int) and l > 0 for l in recurrent_lengths), 'recurrent lengths must be a sequence of positive integers'
        assert sum(recurrent_lengths) == max_len, f'sum of recurrent lengths ({sum(recurrent_lengths)}) must equal max_len ({max_len})'

        boundaries = set(accumulate(recurrent_lengths))
        is_boundary = lambda idx: idx in boundaries

        filter_fn = default(filter_fn, identity)
        filter_kwargs = default(filter_kwargs, {})

        # handle maybe 1d prompt

        prompt, inverse_pack = pack_with_inverse(prompt, '* n')

        prompt = prompt.to(self.device)

        prompt_len = prompt.shape[-1]
        sample_num_times = max(0, max_len - prompt_len)

        if sample_num_times == 0:
            return inverse_pack(prompt[:, :0])

        out = []

        # catch up memories with prompt

        prompt_recurrent_lengths = slice_recurrent_lengths(recurrent_lengths, prompt_len)

        step_out, memories = self(prompt, recurrent_lengths = prompt_recurrent_lengths, update_state = is_boundary)

        # sample first token

        logits = step_out[:, -1]
        filtered_logits = filter_fn(logits, **filter_kwargs)
        sampled = gumbel_sample(filtered_logits, temperature = temperature)
        sampled = rearrange(sampled, 'b -> b 1')

        out.append(sampled)

        # generate remaining tokens step-by-step with recurrent state

        for token_idx in range(sample_num_times - 1):
            total_index = prompt_len + 1 + token_idx

            update_state = is_boundary(total_index)

            step_out, memories = self(sampled, memories = memories, recurrent_lengths = (1,), update_state = update_state)

            logits = step_out[:, -1]
            filtered_logits = filter_fn(logits, **filter_kwargs)
            sampled = gumbel_sample(filtered_logits, temperature = temperature)
            sampled = rearrange(sampled, 'b -> b 1')

            out.append(sampled)

        # concat all newly generated tokens

        out = cat(out, dim = -1)

        return inverse_pack(out)

    def forward(
        self,
        tokens,
        memories: RLTMemories | None = None,
        return_loss = False,
        return_loss_breakdown = False,
        recurrent_lengths: int | Sequence[int] | None = None,
        update_state: bool | None = None
    ):
        # embed

        if self.has_num_tokens:

            if return_loss and not self.has_next_latent_loss:
                tokens, labels = tokens[:, :-1], tokens[:, 1:]

            raw_tokens = tokens
            tokens = self.token_emb(tokens)

        batch, seq_len = tokens.shape[:2]

        # memories

        enc_memories, dec_memories = default(memories, (None, None))
        enc_memories, prev_keys_values = default(enc_memories, (None, None))
        state, dec_memories = default(dec_memories, (None, None))

        prev_num_tokens = prev_keys_values[0].shape[-2] if exists(prev_keys_values) else 0

        # recurrent lengths

        block_size = default(recurrent_lengths, self.recurrent_block_size)
        is_custom = not isinstance(block_size, int)

        recurrent_lengths = to_tuple(block_size) if is_custom else slice_recurrent_lengths(block_size, seq_len, prev_num_tokens)

        assert all(isinstance(l, int) and l > 0 for l in recurrent_lengths), 'recurrent lengths must be a sequence of positive integers'
        assert sum(recurrent_lengths) == seq_len, f'sum of recurrent lengths ({sum(recurrent_lengths)}) must equal sequence length ({seq_len})'

        is_boundary = (lambda _: True) if is_custom else (lambda idx: divisible_by(idx, block_size))

        # maybe initial state

        if not exists(state):
            state = repeat(self.initial_state, 'd -> b 1 d', b = batch)

        # encode

        encoder_out = self.encoder(
            tokens,
            memories = enc_memories,
            return_memories = True,
            return_hiddens = self.attn_residual_cross_encoder
        )

        encoded, next_enc_memories = encoder_out.tokens, encoder_out.memories
        encoder_hiddens = encoder_out.hiddens if self.attn_residual_cross_encoder else []

        # keys and values for cross attention

        keys, values = self.to_encoded_key_values(encoded).chunk(2, dim = -1)
        keys, values = (self.split_heads(t) for t in (keys, values))

        if exists(prev_keys_values):
            prev_keys, prev_values = prev_keys_values
            keys = cat((prev_keys, keys), dim = -2)
            values = cat((prev_values, values), dim = -2)

        # the main proposal, make the decoder of the YOCO setup looped

        decoder_outputs = []
        curr = 0

        for block_len in recurrent_lengths:
            encoded_block = encoded[:, curr : curr + block_len]

            # encoder hiddens, sliced to the current block, to be fed into the attention residual of the decoder

            step_encoder_hiddens = [h[:, curr : curr + block_len] for h in encoder_hiddens]

            curr += block_len
            total_index = prev_num_tokens + curr

            # combine encoded block with state (relies on broadcasting)

            decoder_block = self.combine_encoded_token_and_state(state, encoded_block)

            # slice the encoded key / values to the current sequence position, and map each decoder layer to its encoded layer

            step_keys, step_values = tree_map_tensor(
                lambda t: t[list(self.layer_to_layer_mapping)],
                (keys[..., :total_index, :], values[..., :total_index, :])
            )

            step_keys_values = tuple(zip(step_keys, step_values))

            decoder_output, dec_memories = self.decoder(
                decoder_block,
                keys_values = step_keys_values,
                memories = dec_memories,
                return_memories = True,
                block_outputs = [*step_encoder_hiddens, decoder_block] if self.attn_residual else None
            )

            # append for output

            decoder_outputs.append(decoder_output)

            # set next state as decoder output only at a block boundary

            should_update_state = update_state(total_index) if callable(update_state) else default(update_state, is_boundary(total_index))

            if should_update_state:
                state = decoder_output[:, -1:]

            # truncated bptt

            if exists(self.tbptt_step_size) and divisible_by(total_index, self.tbptt_step_size):
                state, dec_memories = tree_map_detach((state, dec_memories))

        decoded = cat(decoder_outputs, dim = 1)

        memories = RLTMemories(
            EncoderMemories(next_enc_memories, (keys, values)),
            DecoderMemories(state, dec_memories)
        )

        if not self.has_num_tokens:
            assert not return_loss
            return decoded, memories

        logits = self.to_logits(decoded)

        if not return_loss:
            return logits, memories

        if not self.has_next_latent_loss:
            cross_entropy_loss = F.cross_entropy(rearrange(logits, 'b n v -> b v n'), labels, ignore_index = -1)

            if not return_loss_breakdown:
                return cross_entropy_loss

            return cross_entropy_loss, Losses(cross_entropy_loss, None, None)

        # next-latent prediction loss (Teoh et al. https://arxiv.org/abs/2511.05963)

        labels = raw_tokens[:, 1:]
        teacher_logits = logits[:, :-1]

        cross_entropy_loss = F.cross_entropy(
            rearrange(teacher_logits, 'b n v -> b v n'),
            labels,
            ignore_index = -1
        )

        next_latent_loss, kl_div_loss = self.next_latent_prediction.rollout_loss(
            decoded,
            tokens,
            teacher_logits,
            self.to_logits,
            labels,
            kl_loss_weight = self.next_latent_kl_loss_weight
        )

        total_loss = (
            cross_entropy_loss +
            next_latent_loss * self.next_latent_loss_weight +
            kl_div_loss * self.next_latent_kl_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, Losses(cross_entropy_loss, next_latent_loss, kl_div_loss)
