import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, max_len=50, embed_dim=64):
        super().__init__()

        self.embed = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )

        self.pos = nn.Parameter(torch.randn(1, max_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            batch_first=True,
            dim_feedforward=4 * embed_dim,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        x = self.embed(x) + self.pos[:, :x.size(1)]

        x = self.encoder(x, src_key_padding_mask=mask)

        # CLS-token pooling
        cls_rep = x[:, 0]   # (batch, embed_dim)

        return self.fc(cls_rep).squeeze(1)  # logits
