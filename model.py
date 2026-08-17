import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask):
        h = self.norm1(x)

        attention_out, _ = self.attention(
            h,
            h,
            h,
            attn_mask=causal_mask,
            need_weights=False,
        )

        x = x + attention_out
        x = x + self.ffn(self.norm2(x))

        return x


class NameTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embed_dim,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(
            embed_dim,
            vocab_size,
        )

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape

        positions = torch.arange(
            seq_len,
            device=idx.device,
        )

        x = (
            self.token_embedding(idx)
            + self.position_embedding(positions)
        )

        # Upper-triangular causal mask.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=idx.device,
            ),
            diagonal=1,
        ).bool()

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.norm(x)

        logits = self.head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss
