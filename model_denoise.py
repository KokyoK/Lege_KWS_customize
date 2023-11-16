import torch
import torch.nn as nn
import torch.nn.functional as F
class DenoiseNet(nn.Module):
    def __init__(self):
        super(DenoiseNet, self).__init__()
        # 编码器
        self.encoder_conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.encoder_conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)

        # 解码器
        self.decoder_conv1 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder_conv2 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.decoder_conv3 = nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # 编码器
        skip1 = x = F.relu(self.encoder_conv1(x))
        skip2 = x = F.relu(self.encoder_conv2(x))
        x = F.relu(self.encoder_conv3(x))

        # 解码器
        x = F.relu(self.decoder_conv1(x))
        x = F.relu(self.decoder_conv2(x + skip2))  # 跳跃连接
        x = self.decoder_conv3(x + skip1)  # 跳跃连接
        return x

# 实例化模型
if __name__ == "__main__":
    model = DenoiseNet()
    input = torch.randn(1, 1, 16000)
    output = model(input)
