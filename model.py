import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, use_positional_encoding=True):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.position_embedding = nn.Embedding(max_len, embed_dim)
        else:
            self.register_parameter("position_embedding", None)

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        if not self.use_positional_encoding:
            return x

        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        return x + self.position_embedding(positions)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        head_dim = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

        if mask is not None:
            key_mask = mask[:, None, None, :].bool()
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)
        return output, attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.embed_dim)

    def forward(self, x, mask=None):
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))

        context, attention = self.attention(query, key, value, mask=mask)
        context = self._merge_heads(context)
        output = self.out_proj(context)
        return self.dropout(output), attention


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = PositionWiseFeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        attention_output, attention = self.self_attention(x, mask=mask)
        x = self.norm1(x + attention_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x, attention


class MiniTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size=5,
        max_len=20,
        embed_dim=64,
        ff_dim=128,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        use_positional_encoding=True,
        pooling="first",
        num_classes=2,
    ):
        super().__init__()
        if pooling not in {"first", "mean"}:
            raise ValueError("pooling must be either 'first' or 'mean'")
        self.pooling = pooling
        self.embedding = TokenAndPositionEmbedding(
            vocab_size=vocab_size,
            max_len=max_len,
            embed_dim=embed_dim,
            use_positional_encoding=use_positional_encoding,
        )
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def masked_mean_pool(self, x, mask):
        mask = mask.unsqueeze(-1).type_as(x)
        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(self, tokens, mask):
        x = self.embedding(tokens)
        x = self.dropout(x)

        attentions = []
        for layer in self.layers:
            x, attention = layer(x, mask=mask)
            attentions.append(attention)

        if self.pooling == "first":
            pooled = x[:, 0]
        else:
            pooled = self.masked_mean_pool(x, mask)
        logits = self.classifier(pooled)
        return logits
