from argparse import Namespace
import thop
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, include_bn=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.include_bn = include_bn
        if include_bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.include_bn:
            x = self.bn(x)
        return x

class CustomBlock(nn.Module):
    """Custom block matching the provided schematic with depthwise separable convolutions."""
    def __init__(self, channels):
        super().__init__()
        self.dw_conv1 = DepthwiseSeparableConv(channels, channels, (1, 3), padding=(0, 1))
        self.conv1 = DepthwiseSeparableConv(channels, channels * 4, (1, 1), include_bn=False)
        self.conv2 = DepthwiseSeparableConv(channels, channels * 4, (1, 1), include_bn=False)
        self.relu6 = nn.ReLU6(inplace=True)
        self.conv_combined = DepthwiseSeparableConv(channels * 4, channels, (1, 1), include_bn=True)
        self.dw_conv2 = DepthwiseSeparableConv(channels, channels, (1, 3), padding=(0, 1), include_bn=True)

    def forward(self, x):
        x = self.dw_conv1(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x_combined = self.relu6(x1 + x2)
        x_combined = self.conv_combined(x_combined)
        x_out = self.dw_conv2(x_combined)
        return x_out

class StarNet(nn.Module):
    """Network with two CustomBlocks and a final classifier."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = CustomBlock(40)
        self.block2 = CustomBlock(40)
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(40, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.final_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.rand(2, 40, 1, 101)
    # TCResNet8 test
    model =  StarNet(num_classes=10)

    result_tcresnet8 = model(x)
    print(result_tcresnet8.shape)


    args=Namespace()
    args.orth_loss = "no"
    args.denoise = "no"
    device = "cpu"
    starnet = StarNet()
    macs, params = thop.profile(starnet,inputs=(x,))
    print(f"STAR NET macs {macs}, params {params}")