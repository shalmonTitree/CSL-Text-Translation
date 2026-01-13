import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[: x.size(1)]


# -------------------------
# Bottleneck Transformer Encoder
# -------------------------
class BottleneckEncoder(nn.Module):
    def __init__(
        self,
        input_dim=225,
        d_model=256,
        num_layers=4,
        num_heads=8,
        bottleneck_k=32,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # learnable bottleneck queries
        self.bottleneck_q = nn.Parameter(
            torch.randn(bottleneck_k, d_model)
        )

        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )

        # smoothing over bottleneck tokens
        self.smoothing = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats, feat_mask=None, mask_prob=0.0):
        """
        feats: (B, T, 225)
        feat_mask: (B, T) 1 = valid
        """
        B, T, _ = feats.shape

        x = self.input_proj(feats)
        x = self.pos(x)

        if feat_mask is not None:
            key_padding_mask = feat_mask == 0
        else:
            key_padding_mask = None

        enc = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # repeat bottleneck queries for batch
        q = self.bottleneck_q.unsqueeze(0).expand(B, -1, -1)

        # bottleneck attention
        bottleneck, _ = self.cross_attn(
            q, enc, enc, key_padding_mask=key_padding_mask
        )

        # ---------- iterative masking ----------
        if self.training and mask_prob > 0:
            mask = (
                torch.rand(bottleneck.size(0), bottleneck.size(1), device=bottleneck.device)
                < mask_prob
            )
            bottleneck = bottleneck.masked_fill(mask.unsqueeze(-1), 0.0)

        # ---------- smoothing ----------
        b = bottleneck.transpose(1, 2)  # (B, D, K)
        b = self.smoothing(b)
        b = b.transpose(1, 2)

        return self.norm(b)  # (B, K, D)


# -------------------------
# Full CSLT encoder → T5 bridge
# -------------------------
class CSLTBottleneckModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BottleneckEncoder()
        self.to_t5 = nn.Linear(256, 512)

    def forward(self, feats, feat_mask=None, mask_prob=0.15):
        b = self.encoder(feats, feat_mask, mask_prob)
        return self.to_t5(b)  # (B, K, 512)