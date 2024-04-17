import torch
import torch.nn as nn
import torch.nn.functional as F
import speech_dataset as sd
import noisy_dataset as nsd 
torch.manual_seed(42)
import thop
import unet
import time
from mamba import simple_mamba
from ptflops import get_model_complexity_info
# Pytorch implementation of Temporal Convolutions (TC-ResNet).
# Original code (Tensorflow) by Choi et al. at https://github.com/hyperconnect/TC-ResNet/blob/master/audio_nets/tc_resnet.py
#
# Input data represents frequencies (MFCCs) in different channels.
#                      _________________
#                     /                /|
#               freq /                / /
#                   /_______________ / /
#                1 |_________________|/
#                          time


class SiameseTCResNet(nn.Module):
    def __init__(self, k, n_mels, n_classes, n_speaker,args):
        super(SiameseTCResNet, self).__init__()
        self.args = args
        # 使用TCResNet8作为子网络
        if args.backbone =="res":
            self.network = TCResNet8(k, n_mels, n_classes, n_speaker,args)
        else:
            self.network = unet.StarNet(args=args)
        # self.network = unet.StarNet()
        if args.denoise_net == "mamba":
            self.denoise_net = simple_mamba.Mamba()
        elif args.denoise_net == "unet":
            self.denoise_net = unet.UNet()
        elif args.denoise_net == "sub":
            self.denoise_net = Sub()
            
        # self.denoise_net = simple_mamba.Mamba()
        # self.denoise_net = unet.StarNet()
        self.denoised_anchor = None
    def forward(self, anchor,pos,neg):
        if self.args.denoise_loss == "yes":
            anchor = self.denoise_net(anchor)
            pos = self.denoise_net(pos)
            neg = self.denoise_net(neg)
        self.denoised_anchor = anchor
        # 分别处理3个输入
        out_k1, out_s1, map_k1, map_s1 = self.network(anchor)
        out_k2, out_s2, map_k2, map_s2 = self.network(pos)
        out_k3, out_s3, map_k3, map_s3 = self.network(neg)
        return out_k1, map_s1, map_s2, map_s3

    def save(self, is_onnx=0, name="SimTCResNet8"):

        torch.save(self.state_dict(), "saved_model/" + name)

    def load(self, name="SimTCResNet8"):
        self.load_state_dict(torch.load("saved_model/" + name, map_location=lambda storage, loc: storage))

    def set_args(self, args):
        self.args = args
        for name, module in self.__dict__.items():
            if isinstance(module, nn.Module):
                module.set_args(args)
class S2_Block(nn.Module):
    """ S2 ConvBlock using Depth-wise and Point-wise Convolution """

    def __init__(self, in_ch, out_ch):
        super(S2_Block, self).__init__()

        # Convolution for the first layer
        self.conv0 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 9), stride=2,
                                  padding=(0, 4), bias=True)
        self.bn0 = nn.BatchNorm2d(out_ch, affine=True)

        # Convolution for the second layer
        self.conv1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 9), stride=1,
                                  padding=(0, 4),  bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch, affine=True)

        # Residual depth-wise convolution
        self.conv_res = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 9), stride=2,
                                     padding=(0, 4), bias=True)
        self.bn_res = nn.BatchNorm2d(out_ch, affine=True)

    def forward(self, x):
        # First convolution
        out = self.conv0(x)
        out = self.bn0(out)
        out = F.relu(out)
        # Second convolution
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        # Residual connection
        identity = self.conv_res(x)
        identity = self.bn_res(identity)
        identity = F.relu(identity)

        out = out + identity
        out = F.relu(out)

        return out


class TCResNet8(nn.Module):
    """ TC-ResNet8 using Depth-wise and Point-wise Convolution """

    def __init__(self, k, n_mels, n_classes, n_speaker,args):
        super(TCResNet8, self).__init__()
        self.args = args
        # First Convolution layer (Depth-wise and Point-wise)
        self.dw_conv_block = nn.Conv2d(in_channels=n_mels, out_channels=int(16 * k), kernel_size=(1, 3),
                                       padding=(0, 1), bias=True)
        # self.pw_conv_block = nn.Conv2d(in_channels=n_mels, out_channels=int(16 * k), kernel_size=1, bias=False)

        # S2 Blocks
        self.s2_block0 = S2_Block(int(16 * k), int(24 * k))
        self.s2_block0_speaker = S2_Block(int(16 * k), int(24 * k))

        self.s2_block1 = S2_Block(int(24 * k), int(32 * k))
        self.s2_block1_speaker = S2_Block(int(24 * k), int(32 * k))

        self.s2_block2 = S2_Block(int(32 * k), int(48 * k))
        self.s2_block2_speaker = S2_Block(int(32 * k), int(48 * k))

        # Average Pooling and Fully Connected Layer
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 13), stride=1)
        self.fc = nn.Conv2d(in_channels=int(48 * k), out_channels=n_classes, kernel_size=1, padding=0, bias=True)
        self.fc_s = nn.Conv2d(in_channels=int(48 * k), out_channels=n_speaker, kernel_size=1, padding=0, bias=True)

        # Parameters for orthogonal loss
        self.d = 48  # feature数
        # self.share_para = nn.Parameter(torch.randn(
        #     self.d, self.d), requires_grad=True)
        # self.kws_para = nn.Parameter(torch.randn(
        #     self.d, self.d), requires_grad=True)
        # self.speaker_para = nn.Parameter(torch.randn(
        #     self.d, self.d), requires_grad=True)
        self.w_kk = nn.Parameter(torch.randn(
            self.d, self.d), requires_grad=True)
        self.w_ks = nn.Parameter(torch.randn(
            self.d, self.d), requires_grad=True)
        self.w_ss = nn.Parameter(torch.randn(
            self.d, self.d), requires_grad=True)
        self.w_sk = nn.Parameter(torch.randn(
            self.d, self.d), requires_grad=True)
        self.w_s_dis = nn.Parameter(torch.randn(
            self.d, self.d), requires_grad=True)
        self.w_k_dis = nn.Parameter(torch.randn(
            self.d, self.d), requires_grad=True)
        # self.conv_ks = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(1, 3),
        #                                padding=(1, 1), bias=True)
        # self.conv_sk = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(1, 3),
        #                                padding=(1, 1), bias=True)
        self.kws_attn = nn.MultiheadAttention(embed_dim=51, num_heads=1)
        self.sid_attn = nn.MultiheadAttention(embed_dim=51, num_heads=1)
        self.attn_k_weights = None
        self.attn_s_weights = None
        self.test = nn.Linear(48,48)

    def forward(self, x):
        # First depth-wise and point-wise convolution
        out = self.dw_conv_block(x)
        # out = self.pw_conv_block(out)

       # S2 Blocks处理
        share_map = self.s2_block0(out)
        share_map = self.s2_block1(share_map)
        # print( attn_k.shape)


        # keyword recognition path
        # out_k = self.s2_block1(share_map)
        k_map = self.s2_block2(share_map)
        # k_map_T = k_map.squeeze(2).permute(1, 0, 2)

        # speaker recognition 
        # out_s = self.s2_block1_speaker(share_map)
        s_map = self.s2_block2_speaker(share_map)
        # s_map_T = s_map.squeeze(2).permute(1, 0, 2)   

        k_map = self.avg_pool(k_map).squeeze(2,3)
        s_map = self.avg_pool(s_map).squeeze(2,3)
        # todo: use cross ortho k_map -> kk, ks.  s_map -> ss, sk
        # baseline 
        # k_map = k_map.unsqueeze(2).unsqueeze(3)
        # s_map = s_map.unsqueeze(2).unsqueeze(3)
        # ###
        # kk = F.linear(k_map, self.w_kk)
        # ks = F.linear(k_map, self.w_ks)
        # sk = F.linear(s_map, self.w_sk)
        # ss = F.linear(s_map, self.w_ss)
        # k_map = kk.unsqueeze(2).unsqueeze(3)
        # s_map = ss.unsqueeze(2).unsqueeze(3)
        
        # better
        if self.args.orth_loss == "yes":       
            kk = self.w_kk @ k_map.T
            ks = self.w_ks @ k_map.T
            ss = self.w_ss @ s_map.T
            sk = self.w_sk @ s_map.T
            k_map = kk.T.unsqueeze(2).unsqueeze(3)
            s_map = ss.T.unsqueeze(2).unsqueeze(3)
        else:
            k_map = k_map.unsqueeze(2).unsqueeze(3)
            s_map = s_map.unsqueeze(2).unsqueeze(3)

        
        # print(k_map.shape)
        # kws after att
        # out_k = self.avg_pool(k_map)
        out_k = self.fc(k_map)
        out_k = F.softmax(out_k, dim=1)
        out_k = out_k.view(out_k.shape[0], -1)
        
        # speaker after att
        # out_s = self.avg_pool(s_map)
        out_s = self.fc_s(s_map)
        out_s = F.softmax(out_s, dim=1)
        out_s = out_s.view(out_s.shape[0], -1)
    
        return out_k, out_s, k_map, s_map

    def save(self, is_onnx=0, name="TCResNet8"):
        if (is_onnx):
            dummy_input = torch.randn(16, 40, 1, 101)
            torch.onnx.export(self, dummy_input, "TCResNet8.onnx", verbose=True, input_names=["input0"],
                              output_names=["output0"])
        else:
            torch.save(self.state_dict(), "saved_model/" + name)

    def load(self, name="TCResNet8"):
        self.load_state_dict(torch.load("saved_model/" + name, map_location=lambda storage, loc: storage))

import torch
import torch.nn as nn

class Sub(nn.Module):
    def __init__(self, noise_estimation_length=10, alpha=1.0):
        """
        初始化谱减法模块。

        参数:
        noise_estimation_length: 用于噪声估计的时间帧数量，默认为10。
        alpha: 噪声谱的放大系数，默认为1.0。
        """
        super(Sub, self).__init__()
        self.noise_estimation_length = noise_estimation_length
        self.alpha = alpha

    def forward(self, stft_speech):
        """
        前向传播，执行谱减法去噪。

        参数:
        stft_speech: 含噪语音的STFT谱，形状为 [batch, 1, frequency, time]。

        返回:
        去噪后的语音信号的STFT谱。
        """
        # 估计噪声
        noise_estimate = self.estimate_noise(stft_speech, self.noise_estimation_length)
        
        # 执行谱减法
        subtracted_spectrum = stft_speech - self.alpha * noise_estimate
        subtracted_spectrum_clipped = torch.clamp(subtracted_spectrum, min=0)  # 确保谱值非负
        
        return subtracted_spectrum_clipped

    def estimate_noise(self, stft_speech, noise_estimation_length):
        """
        从含噪语音的STFT中估计噪声。
        
        参数:
        stft_speech: 含噪语音的STFT谱，形状为 [batch, 1, frequency, time]。
        noise_estimation_length: 用于噪声估计的时间帧数量。
        
        返回:
        噪声的估计STFT谱，形状为 [batch, 1, frequency, time]。
        """
        # 取语音开头部分的平均作为噪声估计
        noise_estimate = torch.mean(stft_speech[:, :, :, :noise_estimation_length], dim=-1, keepdim=True)
        # 将噪声估计扩展到与语音相同的时间长度
        noise_estimate = noise_estimate.expand(-1, -1, -1, stft_speech.size(-1))
        return noise_estimate


if __name__ == "__main__":
    ROOT_DIR = "dataset/google_noisy/NGSCD/"
    WORD_LIST = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    SPEAKER_LIST = nsd.fetch_speaker_list(ROOT_DIR, WORD_LIST)
    print("num speaker: ", len(SPEAKER_LIST))
    x = torch.rand(1, 40, 1, 101)
    model_fp32 = SiameseTCResNet(k=1, n_mels=40, n_classes=len(WORD_LIST), n_speaker=len(SPEAKER_LIST))
    out = model_fp32(x,x,x)
    
   
    tcres = TCResNet8(k=1, n_mels=40, n_classes=len(WORD_LIST), n_speaker=len(SPEAKER_LIST))
    macs, params = thop.profile(tcres,inputs=(x,))
    print(f"macs {macs}, params {params}")
    
    flops, params = get_model_complexity_info(tcres, (40,1,101), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
    
    start = time.time()
    for i in range(1000):
        out = tcres(x)
    end = time.time()
    print("Running time of tcresnet: ", (end-start))
    
    
    # ROOT_DIR = "dataset/lege/"
    # WORD_LIST = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
    # SPEAKER_LIST = sd.fetch_speaker_list(ROOT_DIR, WORD_LIST)
    # loaders = sd.get_loaders( ROOT_DIR, WORD_LIST,SPEAKER_LIST)
    
    # model_fp32 = SiameseTCResNet(k=1, n_mels=40, n_classes=len(WORD_LIST), n_speaker=len(SPEAKER_LIST))
    # for batch_idx, batch in enumerate(loaders[0]):
    #     anchor_batch, positive_batch, negative_batch = batch
    #     anchor_data, anchor_kws_label, _ = anchor_batch
    #     positive_data, _, _ = positive_batch
    #     negative_data, _, _ = negative_batch
        
        
    #     anchor_out_kws, anchor_out_speaker, positive_out_speaker, negative_out_speaker = model_fp32(anchor_data, positive_data, negative_data)
    #     break
        



