import torch
torch.manual_seed(42)
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import speech_dataset as sd
import model as md
import csv
import log_helper
import pandas as pd
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

def evaluate_testset_all(model, test_dataloader,args):
    model.eval()  # Set the model to evaluation mode
    all_scores = []
    all_labels = []
    total_correct_kws = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (anchor_batch, positive_batch, negative_batch) in enumerate(test_dataloader):
            anchor_data, anchor_clean, [anchor_kws_label,_,_,_] = anchor_batch
            positive_data, _, _ = positive_batch
            negative_data, _, _ = negative_batch

            # if train_on_gpu:
                # anchor_data = anchor_data.cuda()
                # positive_data = positive_data.cuda()
                # negative_data = negative_data.cuda()

            anchor_out_kws, anchor_out_speaker, positive_out_speaker, negative_out_speaker = model(anchor_data, positive_data, negative_data)

            scores, labels = calculate_similarity_and_metrics(anchor_out_speaker, positive_out_speaker, negative_out_speaker)
            all_scores.extend(scores)
            all_labels.extend(labels)
            
            total_correct_kws += torch.sum(torch.argmax(anchor_out_kws, dim=1) == anchor_kws_label).item()
            total_samples += anchor_kws_label.size(0)


            
    test_accuracy = total_correct_kws / total_samples * 100
    print(f'Test Accuracy KWS: {test_accuracy:.2f}%' )
    # 计算 ROC 曲线
    fpr, tpr, threshold = roc_curve(all_labels, all_scores)
    fnr = 1 - tpr
    # 找到在 FAR 为 1% 时的 FRR 和阈值
    frr_at_1_idx = np.argmax(fpr >= 0.01)
    frr_at_1 = fnr[frr_at_1_idx]
    threshold_at_1 = threshold[frr_at_1_idx]
    # 找到在 FAR 为 10% 时的 FRR 和阈值
    frr_at_10_idx = np.argmax(fpr >= 0.1)
    frr_at_10 = fnr[frr_at_10_idx]
    threshold_at_10 = threshold[frr_at_10_idx]
    # eer
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]  # Equal Error Rate

    print(f"FRR at  1%: {frr_at_1*100:.4f}% FAR, Threshold at  1% FAR: {threshold_at_1:.4f}",)
    print(f"FRR at 10%: {frr_at_10*100:.4f}% FAR, Threshold at 10% FAR: {threshold_at_10:.4f}",)
    print("EER: {:.4f} % at threshold {:.4f}".format(EER*100, eer_threshold))

def evaluate_testset(model, test_dataloader, args):
    model.eval()  # Set the model to evaluation mode
    noise_stats = {
        'clean': {'kw':[],'scores': [], 'labels': []},
        'seen': {},
        'unseen': {}
    }

    with torch.no_grad():
        for batch_idx, (anchor_batch, positive_batch, negative_batch) in enumerate(test_dataloader):
            anchor_data, anchor_clean, [anchor_kws_label, anchor_speaker_label, noise_type, noise_snr] = anchor_batch
            positive_data, _, _ = positive_batch
            negative_data, _, _ = negative_batch

            anchor_out_kws, anchor_out_speaker, positive_out_speaker, negative_out_speaker = model(
                anchor_data, positive_data, negative_data)

            scores, labels = calculate_similarity_and_metrics(anchor_out_speaker, positive_out_speaker, negative_out_speaker)
            correct_kws = (torch.argmax(anchor_out_kws, dim=1) == anchor_kws_label).int()

      
            # Categorize and collect data
            for i in range(len(correct_kws)):
                snr = noise_snr[i]
                n_type = noise_type[i]
                correct = correct_kws[i]
                score = [scores[2*i],scores[2*i+1]]
                label = [labels[2*i],labels[2*i+1]]
                # for snr, n_type, correct, score, label in zip(noise_snr, noise_type, correct_kws, scores, labels):
                if snr == '-':
                    category = 'clean'
                    if category not in noise_stats:
                        noise_stats[category] = {'kw':[],'scores': [], 'labels': []}
                    noise_stats[category]['scores'].extend(score)
                    noise_stats[category]['labels'].extend(label)
                    noise_stats[category]['kw'].append(correct)
                    
                else:
                    if 1 <= int(n_type) <= 6:
                        category = 'seen'
                    elif 7 <= int(n_type) <= 8:
                        category = 'unseen'
                    else:
                        continue
                    if snr not in noise_stats[category]:
                        noise_stats[category][snr] = {'kw':[],'scores': [], 'labels': []}
                    noise_stats[category][snr]['scores'].extend(score)
                    noise_stats[category][snr]['labels'].extend(label)
                    noise_stats[category][snr]['kw'].append(correct)
    # print(len(noise_stats['clean']['scores']),
    #             len(noise_stats['clean']['labels']),
    #             len(noise_stats['clean']['kw']))
    # Process data and compute metrics
    results = []
    snr_order= ["-","20","15","10","5","0","-5","-10"]
    for category, data in noise_stats.items():
        if category == 'clean':
            accuracy_kws = sum(data['kw']) / len(data['kw']) * 100
            print(data['labels'][0:10], data['scores'][0:10])
            fpr, tpr, thresholds = roc_curve(data['labels'], data['scores'])
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.abs(fnr - fpr))
            eer = fpr[eer_idx] * 100
            results.append({
                'Category': 'Clean',
                'SNR': '-',
                'Keyword Accuracy (%)': accuracy_kws,
                'EER (%)': eer
            })
        else:
            for snr in snr_order[1:]:
                stats = data[snr]
                accuracy_kws = sum(stats['kw']) / len(stats['kw']) * 100
                fpr, tpr, thresholds = roc_curve(stats['labels'], stats['scores'])
                fnr = 1 - tpr
                eer_idx = np.nanargmin(np.abs(fnr - fpr))
                eer = fpr[eer_idx] * 100
                results.append({
                    'Category': category.capitalize(),
                    'SNR': snr,
                    'Keyword Accuracy (%)': accuracy_kws,
                    'EER (%)': eer
                })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('snr_noise_performance.csv', index=False)
    print("Results saved to 'snr_noise_performance.csv'")

    # Print results for debugging
    for result in results:
        print(f"Category: {result['Category']} SNR: {result['SNR']} - Keyword Accuracy: {result['Keyword Accuracy (%)']}, EER: {result['EER (%)']:.2f}%")

    return df

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
        loss_k = torch.norm(torch.matmul(net.orth_block.w_ss.T, net.orth_block.w_sk), p='fro') **2
        loss_s = torch.norm(torch.matmul(net.orth_block.w_ks.T, net.orth_block.w_kk), p='fro') **2
        

        loss_inner_k = torch.trace(torch.abs(torch.as_tensor(torch.einsum("ij,ij",net.orth_block.ks,net.orth_block.kk), dtype=torch.float32).view(1, 1))) / net.orth_block.w_ss.numel()
        loss_inner_s = torch.trace(torch.abs(torch.as_tensor(torch.einsum("ij,ij",net.orth_block.ss, net.orth_block.sk), dtype=torch.float32).view(1, 1)))/ net.orth_block.w_ss.numel()
        # return loss_k + loss_s + loss_inner_k + loss_inner_s
        # return loss_inner_k + loss_inner_s  #orth
      
        loss = loss_k + loss_s      # cov 
        # loss = torch.Tensor([0])
        return loss

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
        
        
def train(model, num_epochs, loaders,args):
    """
    Trains and validates the model.

    Args:
    model: The PyTorch model to be trained.
    num_epochs: Number of epochs to train for.
    loaders: List of DataLoaders (train, validation, test).
    device: The device to train on ('cuda' or 'cpu').
    """
    logger = log_helper.CsvLogger(filename=args.log,
                                  head=["Epoch","KWS ACC","SV_EER","SV_FRR_1","SV_FRR_10",])

    if train_on_gpu:
        model.to(device)
    [train_dataloader, test_dataloader, dev_dataloader] = loaders
    # [train_dataloader, dev_dataloader, test_dataloader] = loaders

    criterion_kws = nn.CrossEntropyLoss()  # For keyword spotting
    criterion_speaker = nn.TripletMarginLoss(margin=1.0, p=2)  # For speaker identification
    criterion_orth = OrgLoss()  # Custom orthogonal loss
    criterion_noise = DenoiseLoss()


    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    optimizer = torch.optim.Adam([
        {'params': model.network.parameters(), 'lr': 1e-4,'weight_decay':1e-6}, 
        {'params': model.denoise_net.parameters(), 'lr': 1e-4,'weight_decay':1e-6}    
    ])
    prev_kws_acc = 0
    prev_speaker_loss = 999
    prev_EER = 100

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
            if args.denoise_loss == "yes":
                kld_loss = torch.mean(-0.5 * torch.sum(1 + model.denoise_net.log_var - model.denoise_net.mu ** 2 - model.denoise_net.log_var.exp(), dim = 1), dim = 0)
            else:
                kld_loss = 0
            loss_denoise = criterion_noise( model.denoised_anchor,anchor_clean) + 0.1*kld_loss
            loss_kws = criterion_kws(anchor_out_kws, anchor_kws_label)
            loss_speaker = criterion_speaker(anchor_out_speaker, positive_out_speaker, negative_out_speaker)

            loss_orth  =  loss_speaker
            loss = loss_speaker + loss_kws 
            if args.backbone == "decouple":
                loss += model.network.orth_loss
            if args.denoise_loss =="yes":
                loss += 0.01*loss_denoise 
            if args.orth_loss == "yes":
                loss_orth = criterion_orth(model.network)
                loss += loss_orth

            
            # loss_denoise.backward(retain_graph=True)
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
        
        all_scores = []
        all_labels = []
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
                loss_orth = loss_speaker
                # loss_orth = loss_speaker
                loss = loss_kws + loss_speaker + loss_orth

                total_valid_loss += loss.item()
                total_valid_loss_denoise += loss_denoise.item()
                total_valid_loss_kws += loss_kws.item()
                total_valid_loss_speaker += loss_speaker.item()
                total_valid_loss_orth += loss_orth.item()
                total_correct_kws += torch.sum(torch.argmax(anchor_out_kws, dim=1) == anchor_kws_label).item()
                total_samples += anchor_kws_label.size(0)
                
                scores, labels = calculate_similarity_and_metrics(anchor_out_speaker, positive_out_speaker, negative_out_speaker)
                all_scores.extend(scores)
                all_labels.extend(labels)
                

        # 计算 ROC 曲线
        fpr, tpr, threshold = roc_curve(all_labels, all_scores)
        fnr = 1 - tpr
        # 找到在 FAR 为 1% 时的 FRR 和阈值
        frr_at_1_idx = np.argmax(fpr >= 0.01)
        frr_at_1 = fnr[frr_at_1_idx]
        threshold_at_1 = threshold[frr_at_1_idx]
        # 找到在 FAR 为 10% 时的 FRR 和阈值
        frr_at_10_idx = np.argmax(fpr >= 0.1)
        frr_at_10 = fnr[frr_at_10_idx]
        threshold_at_10 = threshold[frr_at_10_idx]
        # eer
        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]  # Equal Error Rate



        avg_valid_loss = total_valid_loss / len(dev_dataloader)
        valid_accuracy = total_correct_kws / total_samples * 100
        print(f"###################################    Epoch {epoch+1}/{num_epochs}     ###########################")
        print(f"FRR at  1%: {frr_at_1*100:.4f}% FAR, Threshold at  1% FAR: {threshold_at_1:.4f}",)
        print(f"FRR at 10%: {frr_at_10*100:.4f}% FAR, Threshold at 10% FAR: {threshold_at_10:.4f}",)
        print("EER: {:.4f} % at threshold {:.4f}".format(EER*100, eer_threshold))
        print("---------------------------------------------------------------------------------")
        print(f' Train Accuracy KWS: {train_accuracy:.2f}%  ｜ Train Loss: {avg_train_loss:.4f}')
        print(f' Valid Accuracy KWS: {valid_accuracy:.2f}%  ｜ Valid Loss: {avg_valid_loss:.4f}')
        print(f'Train Loss KWS: {total_train_loss_kws / len(train_dataloader):.4f}｜ Train Loss Speaker: {total_train_loss_speaker / len(train_dataloader):.4f}｜ Train Loss Orth: {total_train_loss_orth / len(train_dataloader):.4f}')
        print(f'Valid Loss KWS: {total_valid_loss_kws / len(dev_dataloader):.4f}｜ Valid Loss Speaker: {total_valid_loss_speaker / len(dev_dataloader):.4f}｜ Valid Loss Orth: {total_valid_loss_orth / len(dev_dataloader):.4f}')
        # print(f'Total Valid Loss KWS: {total_valid_loss_kws / len(dev_dataloader):.4f}')
        print(f"################################################################")
        logger.log(([
                        f"{epoch}",
                        f"{valid_accuracy:.4f}",
                        f"{EER*100:.4f}",
                        f"{frr_at_1*100:.4f}",
                        f"{frr_at_10*100:.4f}"
                                  ]))
        speaker_loss = total_valid_loss_speaker / len(dev_dataloader)
        if (speaker_loss < prev_speaker_loss or valid_accuracy > prev_kws_acc):
            model.save(name=f"google_noisy/{args.ptname}_{epoch+1}_kwsacc_{valid_accuracy:.2f}_idloss_{speaker_loss:.4f}")
            prev_kws_acc = valid_accuracy
            prev_speaker_loss = speaker_loss
    print("Training complete.")
    
def calculate_similarity_and_metrics(anchor_out_speaker, positive_out_speaker, negative_out_speaker):
    # 计算正样本对的相似度
    pos_distances = torch.norm(anchor_out_speaker - positive_out_speaker, p=2, dim=1)
    neg_distances = torch.norm(anchor_out_speaker - negative_out_speaker, p=2, dim=1)
    
    # 将距离转换为相似度分数（负距离）
    pos_scores = -pos_distances
    neg_scores = -neg_distances

    # 准备标签
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).view(-1).cpu().numpy().tolist()
    scores = torch.cat([pos_scores, neg_scores]).view(-1).cpu().numpy().tolist()
    return scores, labels
# Example usage:
# model = YourModel()
# train_and_validate(model, num_epochs, loaders, device='cuda' if torch.cuda.is_available() else 'cpu')

