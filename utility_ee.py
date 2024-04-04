import torch
torch.manual_seed(42)
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import speech_dataset as sd
import model as md

import numpy as np
from sklearn.metrics import roc_curve

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
time_check = True

# train_on_gpu = False
time_check = False
device = "cuda" if train_on_gpu else "cpu"




if not train_on_gpu:
    print('CUDA is not available.  Using CPU ...')
else:
    print('CUDA is available!  Using GPU ...') 
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_check = True

def calculate_snr(clean_spectrogram, noisy_spectrogram):
    """
    计算信噪比（SNR）

    参数:
        clean_spectrogram (torch.Tensor): 干净语音的时频图
        noisy_spectrogram (torch.Tensor): 含噪声语音的时频图

    返回:
        snr (torch.Tensor): 信噪比（以分贝为单位）
    """
    # 计算噪声的时频图
    noise_spectrogram = noisy_spectrogram - clean_spectrogram

    # 计算信号功率和噪声功率
    signal_power = torch.mean(clean_spectrogram ** 2)
    noise_power = torch.mean(noise_spectrogram ** 2)
    # 避免除以零
    eps = 1e-10
    snr = 10 * torch.log10(signal_power / (noise_power + eps))

    return snr

def evaluate_testset(model, test_dataloader):
    model.eval()  # Set the model to evaluation mode
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (anchor_batch, positive_batch, negative_batch) in enumerate(test_dataloader):
            anchor_data, _, _, = anchor_batch
            positive_data, _, _ = positive_batch
            negative_data, _, _ = negative_batch

            # if train_on_gpu:
                # anchor_data = anchor_data.cuda()
                # positive_data = positive_data.cuda()
                # negative_data = negative_data.cuda()

            _, anchor_out_speaker, positive_out_speaker, negative_out_speaker = model(anchor_data, positive_data, negative_data)

            # 计算说话人之间的相似度分数
            # scores = F.cosine_similarity(anchor_out_speaker, positive_out_speaker)
            scores = -torch.norm(anchor_out_speaker - positive_out_speaker, p=2)

            # 标签为 1 表示同一个说话人，0 表示不同说话人
            all_scores.append(float(scores.cpu().numpy()))
            all_labels.append([1])

            # 你也可以加入 negative pair 的分数和标签
            # _, anchor_out_speaker, _, negative_out_speaker = model(anchor_data, negative_data, negative_data)
            scores = -torch.norm(anchor_out_speaker - negative_out_speaker, p=2)
            all_scores.append(float(scores.cpu().numpy()))
            all_labels.append([0])
    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(all_labels, all_scores)
    fnr = 1 - tpr

    # Find the EER
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]  # Equal Error Rate

    print("EER: {:.4f} at threshold {:.4f}".format(EER, eer_threshold))

# Example usage:
# evaluate_testset(your_model, test_dataloader)

class OrgLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,net):
        # mul = torch.matmul(map_k.squeeze(dim=2), map_s.squeeze(dim=2).permute(0,2,1))
        # o_loss = torch.norm(mul, p='fro') ** 2 / (48*48)
        # o_loss = ((torch.norm(Share_W @ Kws_P.T, p='fro') + torch.norm(Share_W @ Speaker_P.T, p='fro')) ** 2) / (
        #         Share_W.data.shape[0] * Kws_P.data.shape[1])
        # o_loss = ((net.attn_k_weights-net.attn_s_weights) ** 2).mean()
        # return  F.cosine_similarity(net.attn_k_weights.view(-1), net.attn_s_weights.view(-1), dim=0)
        # loss = torch.Tensor([0])
        # if train_on_gpu:
        #     loss.to(device)
        loss_k = torch.trace(torch.abs(torch.as_tensor(torch.einsum("ij,ij",net.w_ss, net.w_sk), dtype=torch.float32).view(1, 1)))
        loss_s = torch.trace(torch.abs(torch.as_tensor(torch.einsum("ij,ij",net.w_ss, net.w_sk), dtype=torch.float32).view(1, 1)))
        return loss_k + loss_s

        # return o_loss * 5
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, clean_audio, denoised_audio):
        mse_loss = torch.mean((clean_audio - denoised_audio) ** 2)
        return mse_loss
   
class DenoiseLoss(nn.Module):
    def __init__(self, n_fft=101, hop_length=1, win_length=101, window='hann'):
        super(DenoiseLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window

    def forward(self, output_spec, target_spec):
        return self.spectral_loss(output_spec, target_spec) + self.phase_loss(output_spec, target_spec) + \
               self.magnitude_loss(output_spec, target_spec)

    def spectral_loss(self, output_spec, target_spec, loss_type='l1'):
        if loss_type == 'l1':
            return F.l1_loss(output_spec, target_spec)
        elif loss_type == 'l2':
            return F.mse_loss(output_spec, target_spec)
        else:
            raise ValueError("Unsupported loss type. Choose 'l1' or 'l2'.")

    def phase_loss(self, output_spec, target_spec):
        return torch.mean(torch.cos(output_spec - target_spec))

    def magnitude_loss(self, output_spec, target_spec, loss_type='l1'):
        if loss_type == 'l1':
            return F.l1_loss(torch.abs(output_spec), torch.abs(target_spec))
        elif loss_type == 'l2':
            return F.mse_loss(torch.abs(output_spec), torch.abs(target_spec))
        else:
            raise ValueError("Unsupported loss type. Choose 'l1' or 'l2'.") 
        
        
def train(model, num_epochs, loaders):
    """
    Trains and validates the model.

    Args:
    model: The PyTorch model to be trained.
    num_epochs: Number of epochs to train for.
    loaders: List of DataLoaders (train, validation, test).
    device: The device to train on ('cuda' or 'cpu').
    """
    if train_on_gpu:
        model.to(device)

    [train_dataloader, dev_dataloader, test_dataloader] = loaders

    criterion_kws = nn.CrossEntropyLoss()  # For keyword spotting
    criterion_speaker = nn.TripletMarginLoss(margin=1.0, p=2)  # For speaker identification
    criterion_orth = OrgLoss()  # Custom orthogonal loss
    criterion_noise = DenoiseLoss()


    # optimizer = torch.optim.Adam(model.network.parameters(), lr=1e-4, weight_decay=0)
    optimizer = torch.optim.Adam([
        {'params': model.network.parameters(), 'lr': 1e-4},  # 第一个损失函数更新 layer1 的参数
        {'params': model.denoise_net.parameters(), 'lr': 1e-3}    # 第二个损失函数更新 layer2 的参数
    ])
    prev_kws_acc = 0
    prev_speaker_loss = 999

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_loss_denoise = 0
        total_train_loss_kws = 0
        total_train_loss_speaker = 0
        total_train_loss_orth = 0
        total_correct_kws = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_dataloader):
            anchor_batch, positive_batch, negative_batch = batch
            anchor_data, anchor_clean, [anchor_kws_label,_,_,_] = anchor_batch
            positive_data, _, _ = positive_batch
            negative_data, _, _ = negative_batch

            anchor_data, anchor_kws_label,anchor_clean = anchor_data.to(device), anchor_kws_label.to(device),anchor_clean.to(device)
            positive_data = positive_data.to(device)
            negative_data = negative_data.to(device)

            optimizer.zero_grad()


            anchor_out_kws, anchor_out_speaker, positive_out_speaker, negative_out_speaker = model(anchor_data, positive_data, negative_data)

            # calculate loss
            loss_denoise = criterion_noise( model.denoised_anchor,anchor_clean)
            loss_kws = criterion_kws(anchor_out_kws, anchor_kws_label)
            loss_speaker = criterion_speaker(anchor_out_speaker, positive_out_speaker, negative_out_speaker)
            loss_orth = criterion_orth(model.network)
            # loss = loss_denoise + loss_kws + loss_speaker + loss_orth
            loss = loss_kws + loss_speaker + loss_orth
            
            loss_denoise.backward(retain_graph=True)
            loss.backward()
            optimizer.step()


            total_train_loss += loss.item()
            total_train_loss_denoise += loss_denoise.item()
            total_train_loss_kws += loss_kws.item()
            total_train_loss_speaker += loss_speaker.item()
            total_train_loss_orth += loss_orth.item()
            total_correct_kws += torch.sum(torch.argmax(anchor_out_kws, dim=1) == anchor_kws_label).item()
            total_samples += anchor_kws_label.size(0)

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}| Step {batch_idx+1}| Loss deoise: {loss_denoise.item():.4f}| Loss KWS: {loss_kws.item():.4f}| Loss Speaker: {loss_speaker.item():.4f}| Loss Orth: {loss_orth.item():.4f}| Total Loss: {loss.item():.4f}| KWS Acc: {100 * torch.sum(torch.argmax(anchor_out_kws, dim=1) == anchor_kws_label).item() / anchor_kws_label.size(0):.2f}%')

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = total_correct_kws / total_samples * 100

        # Validation
        model.eval()
        total_valid_loss = 0
        total_valid_loss_denoise = 0
        total_valid_loss_kws = 0
        total_valid_loss_speaker = 0
        total_valid_loss_orth = 0
        total_correct_kws = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dev_dataloader:
                anchor_batch, positive_batch, negative_batch = batch
                anchor_data, anchor_clean, [anchor_kws_label,_,_,_] = anchor_batch
                positive_data, _, _ = positive_batch
                negative_data, _, _ = negative_batch

                anchor_data, anchor_kws_label, anchor_clean = anchor_data.to(device), anchor_kws_label.to(device), anchor_clean.to(device)
                positive_data = positive_data.to(device)
                negative_data = negative_data.to(device)

                optimizer.zero_grad()

                anchor_out_kws, anchor_out_speaker, positive_out_speaker, negative_out_speaker = model(anchor_data,
                                                                                                       positive_data,
                                                                                                       negative_data)
                # calculate loss
                loss_denoise = criterion_noise(anchor_clean, model.denoised_anchor)
                loss_kws = criterion_kws(anchor_out_kws, anchor_kws_label)
                loss_speaker = criterion_speaker(anchor_out_speaker, positive_out_speaker, negative_out_speaker)
                loss_orth = criterion_orth(model.network)
                loss = loss_denoise + loss_kws + loss_speaker + loss_orth

                total_valid_loss += loss.item()
                total_valid_loss_denoise += loss_denoise.item()
                total_valid_loss_kws += loss_kws.item()
                total_valid_loss_speaker += loss_speaker.item()
                total_valid_loss_orth += loss_orth.item()
                total_correct_kws += torch.sum(torch.argmax(anchor_out_kws, dim=1) == anchor_kws_label).item()
                total_samples += anchor_kws_label.size(0)

        avg_valid_loss = total_valid_loss / len(dev_dataloader)
        valid_accuracy = total_correct_kws / total_samples * 100
        print(f"###################################    Epoch {epoch+1}/{num_epochs}     ###########################")
        print(f' Train Accuracy KWS: {train_accuracy:.2f}%  ｜ Train Loss: {avg_train_loss:.4f}')
        print(f' Valid Accuracy KWS: {valid_accuracy:.2f}%  ｜ Valid Loss: {avg_valid_loss:.4f}')
        print(f'Train Loss Denoise: {total_train_loss_denoise / len(train_dataloader):.4f} | Train Loss KWS: {total_train_loss_kws / len(train_dataloader):.4f}｜ Train Loss Speaker: {total_train_loss_speaker / len(train_dataloader):.4f}｜ Train Loss Orth: {total_train_loss_orth / len(train_dataloader):.4f}')
        print(f'Valid Loss Denoise: {total_valid_loss_denoise / len(dev_dataloader):.4f} | Valid Loss KWS: {total_valid_loss_kws / len(dev_dataloader):.4f}｜ Valid Loss Speaker: {total_valid_loss_speaker / len(dev_dataloader):.4f}｜ Valid Loss Orth: {total_valid_loss_orth / len(dev_dataloader):.4f}')
        # print(f'Total Valid Loss KWS: {total_valid_loss_kws / len(dev_dataloader):.4f}')
        print(f"################################################################")
        speaker_loss = total_valid_loss_speaker / len(dev_dataloader)
        if (speaker_loss < prev_speaker_loss ):
            model.save(name=f"google_noisy/no_denoise_{epoch+1}_kwsacc_{valid_accuracy:.2f}_idloss_{speaker_loss:.4f}")
            prev_kws_acc = valid_accuracy
            prev_speaker_loss = speaker_loss
    print("Training complete.")

# Example usage:
# model = YourModel()
# train_and_validate(model, num_epochs, loaders, device='cuda' if torch.cuda.is_available() else 'cpu')

