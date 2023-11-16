import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

train_on_gpu = torch.cuda.is_available()
device = "cuda" if train_on_gpu else "cpu"

# SincNet Layer
class SincConv(nn.Module):
    def __init__(self, N_filters, kernel_size, sample_rate):
        super(SincConv, self).__init__()

        # Initialize filterbanks
        self.N_filters = N_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Define a window
        self.window = torch.hamming_window(kernel_size).to(device)

        # Frequency cutoffs (learnable)
        self.low_freq = nn.Parameter(torch.Tensor(N_filters)).to(device)
        self.high_freq = nn.Parameter(torch.Tensor(N_filters)).to(device)

        # Initialize cutoff frequencies
        nn.init.uniform_(self.low_freq, 30, 300) # example values
        nn.init.uniform_(self.high_freq, 300, self.sample_rate/2) # example values

    def forward(self, x):
        # Create the sinc filter
        n = torch.linspace(-self.kernel_size//2, self.kernel_size//2, self.kernel_size).to(device)
        n = n.view(1, -1)

        low = 2 * self.low_freq.unsqueeze(1) / self.sample_rate
        high = 2 * self.high_freq.unsqueeze(1) / self.sample_rate

        sinc_low = torch.sin(np.pi * n * low) / (np.pi * n * low)
        sinc_high = torch.sin(np.pi * n * high) / (np.pi * n * high)

        # Handle NaN and replace sinc(0) = 1
        sinc_low[:, self.kernel_size//2] = low.flatten()
        sinc_high[:, self.kernel_size//2] = high.flatten()

        filters = sinc_high - sinc_low
        filters *= self.window
        filters = filters.view(self.N_filters, 1, self.kernel_size)

        # Normalize filters
        filters = filters / torch.sqrt(torch.sum(filters**2, dim=2, keepdim=True))

        return F.conv1d(x, filters)

# AudioNet with SincNet layer
class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()

        # SincNet layer
        self.sinc_conv = SincConv(N_filters=80, kernel_size=251, sample_rate=16000)

        # Additional layers
        self.conv1 = nn.Conv1d(in_channels=80, out_channels=60, kernel_size=5)
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(in_channels=60, out_channels=40, kernel_size=5)
        self.pool2 = nn.MaxPool1d(3)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(101)  # Ensure the temporal dimension is 101

    def forward(self, x):
        x = F.relu(self.sinc_conv(x))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.unsqueeze(2)  # Add an additional dimension to match the target shape [1, 40, 1, 101]
        return x


if __name__ == "__main__":
# Create the model instance
    model = AudioNet()
    # Example input tensor mimicking [1, 16000] shape
    input_tensor = torch.rand(1, 16000)

    # Forward pass to get the output
    output_feature = model(input_tensor)
    print(output_feature.shape)
