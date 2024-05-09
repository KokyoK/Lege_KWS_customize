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

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=(1, 15), padding=(0, 7))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # Ensure no dimensionality change due to odd width
        x = F.avg_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.dp = nn.ConvTranspose2d(in_channels, out_channels, stride = 2, kernel_size=(1, 15), padding=(0, 7))
        # self.conv = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=(1, 15), padding=(0, 7))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 15), padding=(0, 7))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        # Correctly manage the up-sampling to ensure precise dimension recovery
        x = F.interpolate(x, scale_factor=(1, 2), mode='nearest', align_corners=None)
        # x = self.dp(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpecUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder_channels = [40, 24, 48]
        # self.decoder_channels = [48, 24, 40]  # Matches the reverse of encoder outputs

        # self.encoders = nn.ModuleList([EncoderBlock(self.encoder_channels[i], self.encoder_channels[i+1])
        #                                for i in range(len(self.encoder_channels)-1)])
        # self.decoders = nn.ModuleList([DecoderBlock(self.decoder_channels[i], self.decoder_channels[i+1])
        #                                for i in range(len(self.decoder_channels)-1)])
        
        # self.fc_mu = nn.Linear(1200, 32)
        # self.fc_var = nn.Linear(1200, 32)
        # self.decoder_input = nn.Linear(32, 1200)
        # self.log_var = 0
        # self.mu = 0
        
        self.encoder_channels = [40, 80, 160]
        self.decoder_channels = [160, 80, 40]  # Matches the reverse of encoder outputs

        self.encoders = nn.ModuleList([EncoderBlock(self.encoder_channels[i], self.encoder_channels[i+1])
                                       for i in range(len(self.encoder_channels)-1)])
        self.decoders = nn.ModuleList([DecoderBlock(self.decoder_channels[i], self.decoder_channels[i+1])
                                       for i in range(len(self.decoder_channels)-1)])
        
        self.fc_mu = nn.Linear(4000, 128)
        self.fc_var = nn.Linear(4000, 128)
        self.decoder_input = nn.Linear(128, 4000)
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
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
            # Reshape to match the dimensions of the connection exactly for residual connection
            x = F.interpolate(x, size=conn.shape[2:], mode='nearest')
            x = x + conn  # Residual connection by adding
            x = decoder(x)

        # Special handling to ensure the output size matches the input
        # if x.shape[-1] != 101:
        x = F.interpolate(x, size=(1, 101), mode='nearest')
        # return x, mu, log_var
        return x


# Example usage
if __name__ == "__main__":
    model = SpecUNet()
    input_tensor = torch.rand(1, 40, 1, 101)  # Example input tensor (batch_size, channels, height, width)
    output = model(input_tensor)
    print(output[0].shape)  # Output shape should now exactly match the input shape [1, 40, 1, 101]
