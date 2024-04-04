import torch
import torch.nn as nn
import torch.nn.functional as F
import speech_dataset as sd
import noisy_dataset as nsd 
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import thop
torch.manual_seed(42)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 编码器
        self.conv1 = nn.Conv2d(40, 16, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
        self.pool = nn.MaxPool2d((1, 2))

        # 解码器
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=(1, 2), stride=(1, 2))
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(1, 3), padding=(0, 1))
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=(1, 2), stride=(1, 2), output_padding=(0, 1))
        self.conv5 = nn.Conv2d(32, 16, kernel_size=(1, 3), padding=(0, 1))
        self.conv6 = nn.Conv2d(16, 40, kernel_size=(1, 3), padding=(0, 1))

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 编码器
        c1 = self.relu(self.conv1(x))
        p1 = self.pool(c1)
        c2 = self.relu(self.conv2(p1))
        p2 = self.pool(c2)
        c3 = self.relu(self.conv3(p2))

        # 解码器
        up1 = self.relu(self.upconv1(c3))
        merge1 = torch.cat([up1, c2], dim=1)
        c4 = self.relu(self.conv4(merge1))
        up2 = self.relu(self.upconv2(c4))
        merge2 = torch.cat([up2, c1], dim=1)
        c5 = self.relu(self.conv5(merge2))
        c6 = self.conv6(c5)

        return c6

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)
              
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=2, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, [1, 7], 1, (0, 3), groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.g(x)
        x = input + self.drop_path(x)
        return x


class StarNet(nn.Module):
    def __init__(self, base_dim=16, depths=[1, 1, 1, 1], mlp_ratio=2, drop_path_rate=0.0, num_classes=10, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 16
        self.stem = nn.Sequential(
            ConvBN(40, self.in_channel, kernel_size=[1, 3], stride=1, padding=[0, 1]),
            nn.ReLU6()
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, [1, 3], 2, [0, 1])
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# You can adjust the base_dim


    
if __name__ == '__main__':
    unet = UNet()
    starnet = StarNet()
    input_tensor = torch.randn(1, 40, 1, 101)  # batch_size=4
    clean_tensor = torch.randn(1, 40, 1, 101) 

    output_tensor = unet(input_tensor)
    print(output_tensor.shape)
    # flops, params = thop.profile(unet,inputs=(input_tensor,))
    # print(f"U NET flops {flops}, params {params}")
    # print(output_tensor.shape)
    
    # output_tensor = starnet(input_tensor)
    # flops, params = thop.profile(starnet,inputs=(input_tensor,))
    # print(f"STAR NET flops {flops}, params {params}")
    # print(output_tensor.shape)
    # snr = calculate_snr(input_tensor, clean_tensor)
    # print(f'SNR: {snr.item()} dB')

