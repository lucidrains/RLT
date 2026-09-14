from __future__ import annotations
from collections import namedtuple
from functools import partial
from math import ceil
from typing import Callable

import torch
from torch import nn, cat, Tensor
from torch.nn import Module, ModuleList, Linear, Identity, Sequential, RMSNorm
import torch.nn.functional as F

from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange

from torch_einops_utils import tree_map_tensor, temp_eval, pack_with_inverse, tree_map_detach
from rotary_embedding_torch import RotaryEmbedding

# types

RLTMemories = namedtuple('RLTMemories', ['encoder_memories', 'decoder_memories'])
EncoderMemories = namedtuple('EncoderMemories', ['memories', 'keys_values'])
DecoderMemories = namedtuple('DecoderMemories', ['state', 'memories'])
TransformerMemories = namedtuple('TransformerMemories', ['step', 'memories'])

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
    scale = None
):
    assert exists(pt_flex_attention), 'flex_attention requires torch >= 2.5.0'

    rows, cols = q.shape[-2], k.shape[-2]
    single_token = rows == 1

    block_mask = None

    if causal and not single_token:
        prefix_len = cols - rows

        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx + prefix_len >= kv_idx

        block_mask = create_block_mask(mask_mod, B = None, H = None, Q_LEN = rows, KV_LEN = cols, device = q.device)

    return pt_flex_attention(q, k, v, block_mask = block_mask, scale = scale)

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        causal = True,
        cross_attend_key_values = False,
        rotary_embed: RotaryEmbedding | None = None,
        use_flex_attn = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        self.causal = causal and not cross_attend_key_values

        self.norm = RMSNorm(dim)

        self.to_queries = LinearNoBias(dim, dim_inner)
        self.to_key_values = LinearNoBias(dim, dim_inner * 2) if not cross_attend_key_values else None

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = LinearNoBias(dim_inner, dim)

        self.rotary_embed = rotary_embed

        assert not (use_flex_attn and not exists(pt_flex_attention)), 'flex attention is only available on torch 2.5.0 onwards'

        self.use_flex_attn = use_flex_attn

    def forward(
        self,
        tokens,
        keys_values = None,
        memories = None,
        offset = 0,
        return_memories = False
    ):

        tokens = self.norm(tokens)

        q = self.to_queries(tokens)

        # keys and values can be received from the encoder

        if not exists(keys_values):
            assert exists(self.to_key_values), 'keys_values must be provided for cross attention'
            k, v = self.to_key_values(tokens).chunk(2, dim = -1)
            k, v = (self.split_heads(t) for t in (k, v))
        else:
            k, v = keys_values

        q = self.split_heads(q)

        # rotary embedding

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q, offset = offset)
            k = self.rotary_embed.rotate_queries_or_keys(k, offset = offset)

        if exists(memories):
            mk, mv = memories
            k = cat((mk, k), dim = -2)
            v = cat((mv, v), dim = -2)

        rows, cols = q.shape[-2], k.shape[-2]
        single_token = rows == 1

        if self.use_flex_attn:
            out = flex_attention(
                q, k, v,
                causal = self.causal,
                scale = self.scale
            )
        else:
            sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

            if self.causal and not single_token:
                causal_mask = torch.ones((rows, cols), dtype = torch.bool, device = sim.device).triu(cols - rows + 1)
                mask_value = max_neg_value(sim)
                sim = sim.masked_fill(causal_mask, mask_value)

            attn = sim.softmax(dim = -1)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)
        out = self.to_out(out)

        if not return_memories:
            return out

        return out, (k, v)

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
        RMSNorm(dim),
        Linear(dim, dim_inner * 2),
        GEGLU(),
        Linear(dim_inner, dim)
    )

# transformer

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        self_attn = True,
        cross_attn = False,
        self_attn_window_size = None,
        ff_expansion_factor = 4,
        rotary_embed = True,
        dim_rotary = None,
        final_norm = True,
        use_flex_attn = False
    ):
        super().__init__()
        assert not exists(self_attn_window_size) or self_attn_window_size >= 1
        assert not exists(dim_rotary) or dim_rotary <= dim_head

        self.self_attn_window_size = self_attn_window_size

        # rotary embedding

        dim_rotary = default(dim_rotary, dim_head)
        self.rotary_embed = RotaryEmbedding(dim_rotary) if rotary_embed else None

        layers = ModuleList([])

        for _ in range(depth):
            self_attn = Attention(dim = dim, dim_head = dim_head, heads = heads, rotary_embed = self.rotary_embed, use_flex_attn = use_flex_attn) if self_attn else None

            cross_attn = Attention(dim = dim, dim_head = dim_head, heads = heads, cross_attend_key_values = True, use_flex_attn = use_flex_attn) if cross_attn else None

            ff = Feedforward(dim = dim, expansion_factor = ff_expansion_factor)

            layers.append(ModuleList([self_attn, cross_attn, ff]))

        self.layers = layers

        self.norm = RMSNorm(dim) if final_norm else Identity()

    def forward(
        self,
        tokens,
        keys_values = None,
        memories = None,
        return_memories = False
    ):
        seq_len = tokens.shape[-2]

        step, memories = default(memories, (0, []))
        iter_memories = iter(memories)
        next_memories = []

        # layers

        for self_attn, cross_attn, ff in self.layers:

            # self attention

            if exists(self_attn):
                self_attn_out, next_memory = self_attn(tokens, memories = next(iter_memories, None), offset = step, return_memories = True)
                tokens = self_attn_out + tokens

                next_memories.append(next_memory)

            # special cross attention from YOCO

            if exists(cross_attn):
                tokens = cross_attn(tokens, keys_values = keys_values) + tokens

            # feedforward

            tokens = ff(tokens) + tokens

        # norm

        out = self.norm(tokens)

        if not return_memories:
            return out

        # maybe take care of sliding window size - since always doing one token at a time, just do like inference where one slices off the earlier end

        if exists(self.self_attn_window_size):
            w = self.self_attn_window_size
            next_memories = tree_map_tensor(lambda t: t[..., -(w - 1):, :] if w > 1 else t[..., :0, :], next_memories)

        next_step = step + seq_len

        return out, TransformerMemories(next_step, next_memories)

# the recurrent transition they propose

class RecurrentTransition(Module):
    def __init__(
        self,
        dim,
        alpha = 1.
    ):
        super().__init__()
        self.alpha = alpha
        self.norm = RMSNorm(dim)
        self.to_gates = Linear(dim * 2, dim)
        self.to_state = LinearNoBias(dim, dim)

    def forward(
        self,
        state,
        encoded
    ):
        α = self.alpha

        normed_state = self.norm(state)
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
        ff_expansion_factor = 4.,
        recurrent_transition_alpha = 1.,
        dec_sliding_window_size = 16,
        rotary_embed = True,
        dim_rotary = None,
        recurrent_transition: Module | None = None,
        use_flex_attn = False,
        tbptt_step_size: int | None = None
    ):
        super().__init__()
        has_num_tokens = exists(num_tokens)
        self.has_num_tokens = has_num_tokens

        self.token_emb = nn.Embedding(num_tokens, dim) if has_num_tokens else None

        dim_inner = dim_head * heads

        # transformer settings

        transformer_kwargs = dict(heads = heads, dim_head = dim_head, ff_expansion_factor = ff_expansion_factor, rotary_embed = rotary_embed, dim_rotary = dim_rotary, use_flex_attn = use_flex_attn)

        # encoder

        self.encoder = Transformer(dim, depth = enc_depth, **transformer_kwargs)

        # YOCO - Sun et al.

        self.encoded_to_keys_values = LinearNoBias(dim, dim_inner * 2)
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        # recurrence related

        self.tbptt_step_size = tbptt_step_size

        self.initial_state = nn.Parameter(torch.randn(dim) * 1e-2)

        if not exists(recurrent_transition):
            recurrent_transition = RecurrentTransition(dim, alpha = recurrent_transition_alpha)

        self.combine_encoded_token_and_state = recurrent_transition

        # decoder

        self.decoder = Transformer(dim, depth = dec_depth, cross_attn = True, self_attn_window_size = dec_sliding_window_size, **transformer_kwargs)

        # to logits

        self.to_logits = LinearNoBias(dim, num_tokens) if has_num_tokens else None

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
        filter_kwargs: dict = dict(frac_num_tokens = 0.1)
    ):
        assert self.has_num_tokens, '`num_tokens` must be passed to RLT to generate'

        max_len = default(max_len, seq_len)
        assert exists(max_len), 'max_len must be supplied'

        filter_fn = default(filter_fn, identity)
        filter_kwargs = default(filter_kwargs, {})

        # handle maybe 1d prompt

        prompt, inverse_pack = pack_with_inverse(prompt, '* n')

        prompt = prompt.to(self.device)

        sample_num_times = max(0, max_len - prompt.shape[-1])

        if sample_num_times == 0:
            return inverse_pack(prompt[:, :0])

        out = []

        # catch up memories with prompt

        step_out, memories = self(prompt)

        # sample first token

        logits = step_out[:, -1]
        filtered_logits = filter_fn(logits, **filter_kwargs)
        sampled = gumbel_sample(filtered_logits, temperature = temperature)
        sampled = rearrange(sampled, 'b -> b 1')

        out.append(sampled)

        # generate remaining tokens step-by-step with recurrent state

        for _ in range(sample_num_times - 1):
            step_out, memories = self(sampled, memories = memories)

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
    ):
        # embed

        if self.has_num_tokens:

            if return_loss:
                tokens, labels = tokens[:, :-1], tokens[:, 1:]

            tokens = self.token_emb(tokens)

        batch, seq_len = tokens.shape[:2]

        # memories

        enc_memories, dec_memories = default(memories, (None, None))
        enc_memories, prev_keys_values = default(enc_memories, (None, None))
        state, dec_memories = default(dec_memories, (None, None))

        # maybe initial state

        if not exists(state):
            state = repeat(self.initial_state, 'd -> b 1 d', b = batch)

        # encode

        encoded, next_enc_memories = self.encoder(tokens, memories = enc_memories, return_memories = True)

        # keys and values for cross attention

        keys, values = self.encoded_to_keys_values(encoded).chunk(2, dim = -1)
        keys, values = (self.split_heads(t) for t in (keys, values))

        prev_num_tokens = 0

        if exists(prev_keys_values):
            prev_keys, prev_values = prev_keys_values
            prev_num_tokens = prev_keys.shape[-2]

            keys = cat((prev_keys, keys), dim = -2)
            values = cat((prev_values, values), dim = -2)

        # the main proposal, make the decoder of the YOCO setup looped

        decoder_outputs = []

        for index in range(seq_len):

            # one encoded token

            encoded_token = encoded[:, index:index + 1]

            # combine encoded token with state

            decoder_token = self.combine_encoded_token_and_state(state, encoded_token)

            # the keys and values cross attended to must be sliced

            total_index = prev_num_tokens + index + 1

            step_keys_values = (keys[:, :, :total_index], values[:, :, :total_index])

            decoder_output, dec_memories = self.decoder(decoder_token, keys_values = step_keys_values, memories = dec_memories, return_memories = True)

            # append for output

            decoder_outputs.append(decoder_output)

            # set next state as decoder output

            state = decoder_output

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

        loss = F.cross_entropy(rearrange(logits, 'b n v -> b v n'), labels, ignore_index = -1)

        return loss
