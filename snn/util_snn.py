import torch
torch.manual_seed(42)
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import speech_dataset as sd
import torchaudio
import model as md
import csv
import log_helper
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from sklearn.metrics import roc_curve
# from pyroomacoustics.metrics import pesq
from pesq import pesq
import numpy as np

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
device = "cuda" if train_on_gpu else "cpu"




import torch
import torch.nn as nn
import torch.utils.data as data
import snntorch as snn

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
device = "cuda" if train_on_gpu else "cpu"


def rate_based_spike_encoding(real_input, num_steps=100):
    """
    将实值输入转换为脉冲输入，基于 Rate Coding.
    
    Args:
        real_input: 实值输入 [batch_size, num_mels, num_time_steps]
        num_steps: 要生成的脉冲序列的时间步数

    Returns:
        spike_train: 脉冲输入，形状为 [num_steps, batch_size, num_mels, num_time_steps]
    """
    # 将输入的实值归一化到 [0, 1] 范围内
    real_input_normalized = (real_input - real_input.min()) / (real_input.max() - real_input.min())

    # 生成随机数，用于比较生成脉冲
    random_values = torch.rand((num_steps,) + real_input_normalized.shape, device=real_input.device)

    # 根据速率编码规则生成脉冲
    spike_train = (random_values < real_input_normalized.unsqueeze(0)).float()

    return spike_train

# 定义 Activity Regularization
def activity_regularization(spikes, coeff=0.01):
    """
    Penalizes the number of spikes.
    spikes: A list of spikes from different layers and time steps.
    coeff: Regularization coefficient.
    """
    total_spikes = torch.sum(spikes)  # 累积所有时间步的脉冲
    reg_loss = coeff * total_spikes / spikes[0].size(0)  # 根据 batch size 归一化
    return reg_loss

class TimeStepCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(TimeStepCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        计算每个时间步的交叉熵损失并累加
        Args:
            outputs: SNN的输出 [num_steps, batch_size, num_classes]
            targets: 标签 [batch_size]

        Returns:
            loss: 累加的时间步损失
        """
        total_loss = 0
        num_steps = outputs.size(0)  # 获取时间步的数量

        # 对每个时间步计算损失并累加
        for t in range(num_steps):
            total_loss += self.criterion(outputs[t], targets)
        return total_loss / num_steps

def train(model, num_epochs, loaders, args):
    """
    Trains the SNN model with keyword spotting accuracy tracking.

    Args:
    model: The PyTorch SNN model to be trained.
    num_epochs: Number of epochs to train for.
    loaders: List of DataLoaders (train, validation, test).
    args: Arguments for logging, etc.
    """
    logger = log_helper.CsvLogger(filename=args.log,
                                  head=["Epoch", "KWS ACC"])

    if train_on_gpu:
        model.to(device)

    [train_dataloader, test_dataloader, dev_dataloader] = loaders

    # criterion_kws = TimeStepCrossEntropyLoss()  # For keyword spotting
    criterion_kws = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-9)
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_correct_kws = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_dataloader):
            anchor_batch, _, _ = batch
            anchor_data, anchor_clean, [anchor_kws_label, _, _, _] = anchor_batch

             # 使用手动实现的脉冲编码函数，将实值输入转换为脉冲输入
            spike_anchor_data = rate_based_spike_encoding(anchor_data, num_steps=100)
            # Move data to the correct device
            spike_anchor_data, anchor_kws_label = spike_anchor_data.to(device), anchor_kws_label.to(device)
            optimizer.zero_grad()

            # Forward pass through the SNN
            anchor_out_kws = model(spike_anchor_data)
            # print(anchor_out_kws.shape)

            # Loss calculation
            loss_kws = criterion_kws(anchor_out_kws, anchor_kws_label)
            reg_loss = activity_regularization(anchor_out_kws)

            # Backpropagation and optimization
            loss = loss_kws + reg_loss
            loss.backward()
            print(loss)
            optimizer.step()

            total_train_loss += loss_kws.item()

            # Calculate accuracy for keyword spotting
            total_correct_kws += torch.sum(torch.argmax(anchor_out_kws, dim=1) == anchor_kws_label).item()
            # total_correct_kws += torch.sum(torch.argmax(anchor_out_kws.sum(dim=0), dim=1) == anchor_kws_label).item()
            total_samples += anchor_kws_label.size(0)

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}| Step {batch_idx+1}| Loss KWS: {loss_kws.item():.4f}| KWS Acc: {100 * total_correct_kws / total_samples:.2f}%')

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = total_correct_kws / total_samples * 100

        # Log the accuracy after each epoch
        logger.log([f"{epoch}", f"{train_accuracy:.4f}"])
        print(f"Epoch {epoch+1}/{num_epochs} | Train Accuracy: {train_accuracy:.2f}% | Avg Loss: {avg_train_loss:.4f}")

    print("Training complete.")