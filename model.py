import torch
import torch.nn as nn
import torch.nn.functional as F
import speech_dataset as sd


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
    def __init__(self, k, n_mels, n_classes, n_speaker):
        super(SiameseTCResNet, self).__init__()
        # 使用TCResNet8作为子网络
        self.network = TCResNet8(k, n_mels, n_classes, n_speaker)

    def forward(self, anchor,pos,neg):
        # 分别处理两个输入
        out_k1, out_s1, map_k1, map_s1 = self.network(anchor)
        out_k2, out_s2, map_k2, map_s2 = self.network(pos)
        out_k3, out_s3, map_k3, map_s3 = self.network(neg)



        return out_k1, map_s1, map_s2, map_s3

    def save(self, is_onnx=0, name="SimTCResNet8"):

        torch.save(self.state_dict(), "saved_model/" + name)

    def load(self, name="SimTCResNet8"):
        self.load_state_dict(torch.load("saved_model/" + name, map_location=lambda storage, loc: storage))

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

    def __init__(self, k, n_mels, n_classes, n_speaker):
        super(TCResNet8, self).__init__()


        self.dw_conv_block = nn.Conv2d(in_channels=n_mels, out_channels=int(16 * k), kernel_size=(1, 3),
                                       padding=(0, 1), bias=True)
        
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



    def forward(self, x):
        out = self.dw_conv_block(x)

        # S2 Blocks处理
        share_map = self.s2_block0(out)
        # print( attn_k.shape)


        # keyword recognition path
        out_k = self.s2_block1(share_map)
        k_map = self.s2_block2(out_k)

        # speaker recognition 
        out_s = self.s2_block1_speaker(share_map)
        s_map = self.s2_block2_speaker(out_s)
        
        k_map = self.avg_pool(k_map).squeeze()
        s_map = self.avg_pool(s_map).squeeze()

        
        # kws after att
        out_k = self.fc(k_map.unsqueeze(2).unsqueeze(3))
        out_k = F.softmax(out_k, dim=1)
        out_k = out_k.view(out_k.shape[0], -1)
        
        # speaker after att
        out_s = self.fc_s(s_map.unsqueeze(2).unsqueeze(3))
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


if __name__ == "__main__":
    x = torch.rand(1, 40, 1, 101)
    ROOT_DIR = "dataset/lege/"
    WORD_LIST = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
    SPEAKER_LIST = sd.fetch_speaker_list(ROOT_DIR, WORD_LIST)
    loaders = sd.get_loaders( ROOT_DIR, WORD_LIST,SPEAKER_LIST)
    
    model_fp32 = SiameseTCResNet(k=1, n_mels=40, n_classes=len(WORD_LIST), n_speaker=len(SPEAKER_LIST))
    for batch_idx, batch in enumerate(loaders[0]):
        anchor_batch, positive_batch, negative_batch = batch
        anchor_data, anchor_kws_label, _ = anchor_batch
        positive_data, _, _ = positive_batch
        negative_data, _, _ = negative_batch
        
        
        anchor_out_kws, anchor_out_speaker, positive_out_speaker, negative_out_speaker = model_fp32(anchor_data, positive_data, negative_data)
        break
        



