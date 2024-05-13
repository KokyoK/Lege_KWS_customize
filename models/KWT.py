import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class KWT(nn.Module):
    def __init__(self, args, input_dim=40, model_dim=192, nhead=3, num_layers=12, num_classes=10):
        self.args = args
        super(KWT, self).__init__()
        self.proj = nn.Linear(input_dim, model_dim)  # 输入投影层
        encoder_layer = TransformerEncoderLayer(d_model=model_dim, nhead=nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = x.squeeze(2)  # 去掉多余的维度，变为 [batch, 40, 101]
        x = x.permute(2, 0, 1)  # 重新排列维度以适应Transformer [sequence length, batch size, features]
        x = self.proj(x)  # 投影到模型维度
        x = self.transformer(x)  # Transformer处理
        x = x.permute(1, 2, 0)  # 为池化调整维度
        x = self.pool(x).squeeze(-1)  # 平均池化
        x = self.classifier(x)  # 分类层
        return x


