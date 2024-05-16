import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels)

    def forward(self, x):
        x = self.depthwise(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 15), padding=(0, 7))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.avg_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
        return x

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.upsample = PixelShuffleUpsample(in_channels, out_channels, scale_factor=2)
        self.conv = DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=k, padding=(0, 7))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpecUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_channels = [40, 80, 160]
        self.decoder_channels = [160, 80, 40]  # Matches the reverse of encoder outputs

        self.encoders = nn.ModuleList([EncoderBlock(self.encoder_channels[i], self.encoder_channels[i+1], (1, 15))
                                       for i in range(len(self.encoder_channels)-1)])
        self.decoders = nn.ModuleList([DecoderBlock(self.decoder_channels[i], self.decoder_channels[i+1], (1, 15))
                                       for i in range(len(self.decoder_channels)-1)])
        
        d = 4000
        self.fc_mu = nn.Linear(d, 128)
        self.fc_var = nn.Linear(d, 128)
        self.decoder_input = nn.Linear(128, d)
        
        # # 仅进行深度卷积用于跳跃连接
        self.skip_convs = nn.ModuleList([DepthwiseConv2d(self.encoder_channels[2-i], kernel_size=(1, 1), padding=(0, 0))
                                         for i in range(len(self.encoder_channels)-1)])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x):
        connections = []

        for encoder in self.encoders:
            x = encoder(x)
            connections.append(x)
            
        temp = torch.flatten(connections[-1], start_dim=1)   
        self.mu = self.fc_mu(temp)
        self.log_var = self.fc_var(temp)
        temp = self.reparameterize(self.mu, self.log_var)
        temp = self.decoder_input(temp).reshape(connections[-1].shape[0], connections[-1].shape[1], connections[-1].shape[2], connections[-1].shape[3])
        
        for decoder, conn in zip(self.decoders, reversed(connections)):
            x = F.interpolate(x, size=conn.shape[2:], mode='nearest')
            x = x + conn  # Residual connection by adding
            x = decoder(x)

        x = F.interpolate(x, size=(1, 101), mode='nearest')
        return x

# Example usage
if __name__ == "__main__":
    model = SpecUNet()
    input_tensor = torch.rand(1, 40, 1, 101)  # Example input tensor (batch_size, channels, height, width)
    output = model(input_tensor)
    print(output[0].shape)  # Output shape should now exactly match the input shape [1, 40, 1, 101]
