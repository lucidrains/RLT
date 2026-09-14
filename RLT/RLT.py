from __future__ import annotations
from collections import namedtuple
from functools import partial

import torch
from torch import nn, cat
from torch.nn import Module, ModuleList, Linear, Identity, Sequential, RMSNorm
import torch.nn.functional as F

from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange

from torch_einops_utils import tree_map_tensor
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

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        causal = True,
        cross_attend_key_values = False,
        rotary_embed: RotaryEmbedding | None = None
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

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        i, j = sim.shape[-2:]

        if self.causal and i > 1:
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = sim.device).triu(j - i + 1)
            mask_value = max_neg_value(sim)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)
        out = self.to_out(out)

        if not return_memories:
            return out

        return out, [k, v]

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
        final_norm = True
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
            self_attn = Attention(dim = dim, dim_head = dim_head, heads = heads, rotary_embed = self.rotary_embed) if self_attn else None

            cross_attn = Attention(dim = dim, dim_head = dim_head, heads = heads, cross_attend_key_values = True) if cross_attn else None

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
    ):
        super().__init__()
        assert not exists(dec_sliding_window_size) or dec_sliding_window_size >= 1
        assert not exists(dim_rotary) or dim_rotary <= dim_head

        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None

        dim_inner = dim_head * heads

        # encoder

        self.encoder = Transformer(dim, depth = enc_depth, dim_head = dim_head, heads = heads, rotary_embed = rotary_embed, dim_rotary = dim_rotary)

        # YOCO - Sun et al.

        self.encoded_to_keys_values = LinearNoBias(dim, dim_inner * 2)
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        # initial recurrent state (wip)

        self.initial_state = nn.Parameter(torch.randn(dim) * 1e-2)

        self.combine_encoded_token_and_state = RecurrentTransition(dim, alpha = recurrent_transition_alpha)

        # decoder

        self.decoder = Transformer(dim, depth = dec_depth, dim_head = dim_head, heads = heads, cross_attn = True, self_attn_window_size = dec_sliding_window_size, rotary_embed = rotary_embed, dim_rotary = dim_rotary)

        # to logits

        self.to_logits = LinearNoBias(dim, num_tokens) if exists(num_tokens) else None

    def forward(
        self,
        tokens,
        memories: RLTMemories | None = None,
        return_loss = False
    ):
        # embed

        if exists(self.token_emb):

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

        decoded = cat(decoder_outputs, dim = 1)

        memories = RLTMemories(
            EncoderMemories(next_enc_memories, (keys, values)),
            DecoderMemories(state, dec_memories)
        )

        if not exists(self.to_logits):
            assert not return_loss
            return decoded, memories

        logits = self.to_logits(decoded)

        if not return_loss:
            return logits, memories

        loss = F.cross_entropy(rearrange(logits, 'b n v -> b v n'), labels, ignore_index = -1)

        return loss
