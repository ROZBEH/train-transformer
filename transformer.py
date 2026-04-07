import math
import torch
from torch import nn
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class KeyValueCache:
    key: torch.Tensor  # shape: (batch, n_heads, T, d_head)
    value: torch.Tensor  # shape: (batch, n_heads, T, d_head)

    def append(self, new_k: torch.Tensor, new_v: torch.Tensor) -> "KeyValueCache":
        return KeyValueCache(
            key=torch.cat([self.key, new_k], dim=2),
            value=torch.cat([self.value, new_v], dim=2)
        )

    @property
    def length(self) -> int:
        return self.key.size(2)  # Time dimension

@dataclass
class Cache:
    layers: List[Optional[KeyValueCache]] = field(default_factory=list)

    def __post_init__(self):
        if self.layers and any(kv is not None for kv in self.layers):
            lengths = [kv.length for kv in self.layers if kv is not None]
            if len(set(lengths)) != 1:
                raise ValueError("Inconsistent KV lengths across layers.")
            self._length = lengths[0]
        else:
            self._length = 0

    @property
    def length(self) -> int:
        return self._length

    def append(self, new_kvs: List[Optional[KeyValueCache]]) -> "Cache":
        updated = []
        for old_kv, new_kv in zip(self.layers, new_kvs):
            if old_kv is None:
                updated.append(new_kv)
            elif new_kv is None:
                updated.append(old_kv)
            else:
                updated.append(old_kv.append(new_kv.key, new_kv.value))
        return Cache(layers=updated)

    @staticmethod
    def empty(num_layers: int) -> "Cache":
        return Cache([None] * num_layers)

class MultiHeadAttention(nn.Module):
    def __init__(self, dmodel, nheads) -> None:
        super().__init__()
        if dmodel % nheads != 0:
            raise ValueError(f"nheads should divide dmodel evenly: got dmodel={dmodel}, nheads={nheads}")
        
        self.dmodel = dmodel
        self.nheads = nheads
        self.dhead = dmodel // nheads

        self.kv = nn.Linear(dmodel, 2*dmodel)  # project to K, V together
        self.q = nn.Linear(dmodel, dmodel)
        self.dropout_attention = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.projection = nn.Linear(dmodel, dmodel)
    
    def forward(self, x, casual_mask=None, padding_mask=None, layer_cache: Optional[KeyValueCache] = None):
        B, T, _ = x.size()

        kv = self.kv(x)
        
        q = self.q(x)  # shape: (B, T, dmodel)
        Q = q.view(B, T, self.nheads, self.dhead).transpose(1, 2)  # (B, nheads, T, dhead)
        kv = kv.view(B, -1, self.nheads, 2 * self.dhead).transpose(1, 2)  # (B, nheads, T, 2*dhead) <<<<< Note that -1 is a placeholder to tell pytorch to infer the right shape
        K, V = torch.chunk(kv, 2, dim=-1)

        # Scaled dot-product attention
        # Very clever, here you are multiplying (B, nheads, T, dhead) x (B, nheads, dhead, T)
        # which leads to (B, nheads, T, T)
        # you cannot simply use Q@K.T since this is not a 2D multiplication
        # <<<<< note that for the K you have to transpose the last two dimensions
        if layer_cache:
            updated_layer_cache = layer_cache.append(new_k=K, new_v=V)
            K = updated_layer_cache.key
            V = updated_layer_cache.value
            # no need for mask since we are doing 1 token at a time
            casual_mask = padding_mask = None
        else:
            updated_layer_cache = KeyValueCache(key=K, value=V)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dhead)  # (B, nheads, T, T)
        if casual_mask is not None:
            attention_scores = attention_scores + casual_mask
        if padding_mask is not None:
            attention_scores = attention_scores.masked_fill(padding_mask, float('-inf'))
        
        attn_weights = self.softmax(attention_scores)
        attn_weights = self.dropout_attention(attn_weights)
        output = torch.matmul(attn_weights, V)  # (B, nheads, T, dhead)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(B, T, self.dmodel)  # (B, T, dmodel) <<<<< Note that the transpose happens here before the projection layer 
        output = self.projection(output)
        return output, updated_layer_cache

class PositionalEncoding(nn.Module):
    def __init__(self, dmodel, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dmodel, 2) * (-math.log(10000.0) / dmodel)) # <<<< Remember that doing it this way makes the math easier
        pe = torch.zeros(1, max_len, dmodel) # <<< note the 1 along the batch dimension here
        pe[0, :, 0::2] = torch.sin(position * div_term) # <<<< note how is it interleaving via 0::2 or 1::2
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # This tells pytorch that pe is a persistent tensor and not a parameter.
        # It doesn't get updated by backpropagation but it gets saved and loaded into
        # state_dict(). It moves with the model as you call to()
        self.register_buffer('pe', pe)

    def forward(self, x, pos_offset=0):
        return x + self.pe[:, pos_offset:pos_offset+x.size(1)]

class Decoder(nn.Module):
    def __init__(self, dmodel, nheads) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dmodel) # <<<< Note dmodel in this layernorm section
        self.mha = MultiHeadAttention(dmodel=dmodel, nheads=nheads)
        self.ff = nn.Sequential(
            nn.Linear(dmodel, dmodel * 4),
            nn.GELU(), # <<<<< Note this GELU in the MLP section
            nn.Linear(dmodel * 4, dmodel)
        )
        self.dropout_mha = nn.Dropout(p=0.1)
        self.dropout_ff = nn.Dropout(p=0.1)
        self.ln2 = nn.LayerNorm(dmodel)

    def forward(self, x, casual_mask, padding_mask, layer_cache: Optional[KeyValueCache] = None):
        """
        Note that this implementation follows the original transformer implementation which
        uses layernorm after mha/residual and mlp/residual. The GPT implementation uses layernorm
        before mha/residual and also before mlp/residual.
        Original "Attention is all you need":
            x -> (attention or projection block) -> Add(x) -> LayerNorm
        GPT:
            x -> LayerNorm -> (attention or projection block) -> Add(x)
        Using pre layernorm makes deeper models more stable
        """
        x, updated_layer_cache = self.mha(x, casual_mask, padding_mask, layer_cache)
        x = self.dropout_mha(x) + x
        x = self.ln1(x)
        x = self.ff(x)
        x = self.dropout_ff(x) + x
        x = self.ln2(x)
        return x, updated_layer_cache

class Transformer(nn.Module):
    def __init__(self, vocabsize, dmodel, nheads, nlayers, pad_token_id=0) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.nlayers = nlayers
        # adding padding idx since we dont want it to learn
        self.embed = nn.Embedding(num_embeddings=vocabsize, embedding_dim=dmodel, padding_idx=pad_token_id)
        self.pos_enc = PositionalEncoding(dmodel=dmodel)
        self.decoders = nn.ModuleList([Decoder(dmodel, nheads) for _ in range(nlayers)]) # <<<< Note the nn.Module list. Note you cannot make it a sequential module since you are passing multiple argumnets like x and masks
        self.ln_final = nn.LayerNorm(dmodel)
        self.linear = nn.Linear(in_features=dmodel, out_features=vocabsize)
        self.dropout_projection = nn.Dropout(p=0.1)

    def forward(self, x, cache: Optional[Cache] = None):
        pos_offset = cache.length if cache else 0
        new_cache_layers = []
        # x: (batch, seq_len)
        B, T = x.size()
        # create the mask
        # Create padding mask: (batch, seq_len)
        padding_mask = (x == self.pad_token_id).unsqueeze(1).unsqueeze(2)  # <<<< (B, 1, 1, T)  Note the unsqueeze
        padding_mask = padding_mask.expand(B, 1, T, T)  # broadcast to match (B, 1, T, T)

        # Create causal mask: (1, 1, T, T)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        causal_mask = causal_mask.float().masked_fill(causal_mask, float('-inf')).unsqueeze(0).unsqueeze(0) # <<<< note the unsqueezes here

        x = self.embed(x)
        x = self.pos_enc(x, pos_offset)
        for i in range(self.nlayers): # <<<< note that this needs to be a for loop
            layer_cache = None if cache is None else cache.layers[i]
            x, new_cache = self.decoders[i](x, causal_mask, padding_mask, layer_cache)
            new_cache_layers.append(new_cache)
        # x = self.ln_final(x)
        x = self.linear(x)
        x = self.dropout_projection(x)
        updated_cache = Cache(new_cache_layers)
        return x, updated_cache