import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" 

class SE_SPP_Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SE_SPP_Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.spp_decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out)
        spp_out = F.interpolate(self.spp_decoder(enc_out), size=dec_out.shape[2:], mode='bilinear', align_corners=False)
        return dec_out, spp_out

class BCResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BCResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class BCResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=12):
        super(BCResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.linear_s = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BCResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        s_map = out
        out = F.avg_pool2d(out, (out.size(2), out.size(3)))  # Global average pooling
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out,out,s_map, s_map

class SE_SPP_KWS(nn.Module):
    def __init__(self, num_blocks, num_classes=12):
        super(SE_SPP_KWS, self).__init__()
        self.se_spp_net = SE_SPP_Net()
        self.bc_resnet = BCResNet(num_blocks, num_classes)
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()
        self.criterion_bce = nn.BCELoss()
        self.denoised = None
        self.spp = None

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)  # (batch_size, height, channels, width) -> (batch_size, channels, height, width)
        denoised, spp = self.se_spp_net(x)
        self.denoised = denoised
        self.spp = spp
        combined_input = torch.cat((denoised, spp), dim=1)
        outs = self.bc_resnet(combined_input)
        return outs
    def generate_targets(self, spectrograms, threshold=-20):
        target_denoised_list = []
        target_spp_list = []

        for spectrogram in spectrograms:
            log_mel_spec = spectrogram.squeeze(1).cpu().numpy()  # Shape: (n_mels, time_steps)

            # Normalize to [0, 1]
            log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())

            # Create the target denoised output
            target_denoised = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_mels, time_steps)
            target_denoised_list.append(target_denoised)

            # Create the target SPP map (simple thresholding for illustration)
            target_spp = (log_mel_spec > threshold).astype(float)
            target_spp = torch.tensor(target_spp, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_mels, time_steps)
            target_spp_list.append(target_spp)

        # Stack all the targets to form a batch
        target_denoised_batch = torch.stack(target_denoised_list).squeeze(2)  # Shape: (batch_size, 1, n_mels, time_steps)
        target_spp_batch = torch.stack(target_spp_list).squeeze(2)  # Shape: (batch_size, 1, n_mels, time_steps)

        return target_denoised_batch.to(device), target_spp_batch.to(device)

    def calculate_loss(self,  input_spectrograms, lambda_denoising=0.01, lambda_spp=1.0):
        target_denoised_batch, target_spp_batch = self.generate_targets(input_spectrograms)
        # Ensure the target tensors are the same shape as the outputs
        target_denoised = F.interpolate(target_denoised_batch, size=self.denoised.shape[2:], mode='bilinear', align_corners=False).squeeze(2)
        target_spp = F.interpolate(target_spp_batch, size=self.spp.shape[2:], mode='bilinear', align_corners=False).squeeze(2)

      
        loss_mse = self.criterion_mse(self.denoised, target_denoised)
        loss_bce = self.criterion_bce(self.spp, target_spp)
        total_loss =  lambda_denoising * loss_mse + lambda_spp * loss_bce
        return total_loss

if __name__ == "__main__":
    x = torch.rand(2, 40, 1, 101)  # Sample input: (batch_size, height, channels, width)
    y = torch.rand(2, 40, 1, 101)  # Sample input: (batch_size, height, channels, width)
    target_labels = torch.randint(0, 10, (1,))  # Sample target labels

    model = SE_SPP_KWS(num_blocks=[2, 2, 2], num_classes=10)
    output,_,_, s_map = model(x)
    loss = model.calculate_loss(output, target_labels, x)
    print("Total loss:", loss.item())