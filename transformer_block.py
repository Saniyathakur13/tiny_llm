import torch
import torch.nn as nn
import math

embedding_size = 8
block_size = 4
num_heads = 2
head_size = embedding_size // num_heads


class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.query = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.key = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.value = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.projection = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.norm1 = nn.LayerNorm(
            embedding_size
        )

        self.norm2 = nn.LayerNorm(
            embedding_size
        )

        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_size)
        )

    def forward(self, x):

        # x shape:
        # [batch_size, block_size, embedding_size]

        B, T, C = x.shape

        # -------------------------
        # 1. Query, Key, Value
        # -------------------------

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # -------------------------
        # 2. Split into heads
        # -------------------------

        Q = Q.view(
            B,
            T,
            num_heads,
            head_size
        )

        K = K.view(
            B,
            T,
            num_heads,
            head_size
        )

        V = V.view(
            B,
            T,
            num_heads,
            head_size
        )

        # Move heads before tokens

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Shape:
        # [batch, heads, tokens, head_size]

        # -------------------------
        # 3. Attention scores
        # -------------------------

        scores = (
            Q @ K.transpose(-2, -1)
        ) / math.sqrt(head_size)

        # -------------------------
        # 4. Causal mask
        # -------------------------

        mask = torch.tril(
            torch.ones(T, T, device=x.device)
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # -------------------------
        # 5. Attention weights
        # -------------------------

        weights = torch.softmax(
            scores,
            dim=-1
        )

        # -------------------------
        # 6. Attention output
        # -------------------------

        attention = weights @ V

        # -------------------------
        # 7. Combine heads
        # -------------------------

        attention = attention.transpose(1, 2)

        attention = attention.contiguous().view(
            B,
            T,
            C
        )

        attention = self.projection(
            attention
        )

        # -------------------------
        # 8. Residual + LayerNorm
        # -------------------------

        x = self.norm1(
            x + attention
        )

        # -------------------------
        # 9. Feed Forward Network
        # -------------------------

        ffn_output = self.ffn(x)

        # -------------------------
        # 10. Residual + LayerNorm
        # -------------------------

        x = self.norm2(
            x + ffn_output
        )

        return x