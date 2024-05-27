import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, ff_dim=256, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x

class KWT(nn.Module):
    def __init__(self, num_blocks=12, embed_dim=64, num_heads=4, ff_dim=256, dropout_rate=0.1, num_classes=10):
        super(KWT, self).__init__()
        self.conv = nn.Conv2d(1, embed_dim, kernel_size=(40, 1))
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.s_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 13)
        )

    def forward(self, x):
        x = x.permute(0,2,1,3)
        x = self.conv(x)
        x = x.squeeze(2).permute(2, 0, 1)  # (batch_size, embed_dim, time) -> (time, batch_size, embed_dim)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.mean(dim=0)  # Global average pooling
        k_map = self.classification_head(x)
        out_k = F.softmax(k_map, dim=-1)
        s_map = self.s_head(x)
        return out_k, out_k, s_map, s_map




if __name__ == "__main__":
    x = torch.rand(1, 40, 1, 101)
    # TCResNet8 test
    model_tcresnet8 = KWT()
    result_tcresnet8 = model_tcresnet8(x)
    print(result_tcresnet8)
