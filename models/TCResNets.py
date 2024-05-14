import torch
import torch.nn as nn
import torch.nn.functional as F

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


class S1_Block(nn.Module):
    """ S1 ConvBlock used in Temporal Convolutions 
        -> CONV -> BN -> RELU -> CONV -> BN -> (+) -> RELU
        |_______________________________________|
    """
    def __init__(self, out_ch):
        super(S1_Block, self).__init__()

        # First convolution layer
        self.conv0 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 9), stride=1,
                                padding=(0, 4), bias=False)
        self.bn0 = nn.BatchNorm2d(out_ch, affine=True)
        # Second convolution layer
        self.conv1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 9), stride=1,
                                padding=(0, 4), bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, affine=True)

    def forward(self, x):
        identity = x

        out = self.conv0(x)
        out = self.bn0(out)
        out = F.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        
        out += identity
        out = F.relu(out)

        return out


class S2_Block(nn.Module):
    """ S2 ConvBlock used in Temporal Convolutions 
        -> CONV -> BN -> RELU -> CONV -> BN -> (+) -> RELU
        |_______-> CONV -> BN -> RELU ->________|
    """
    def __init__(self, in_ch, out_ch):
        super(S2_Block, self).__init__()

        # First convolution layer
        self.conv0 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 9), stride=2, 
                                padding=(0, 4), bias=True)
        self.bn0 = nn.BatchNorm2d(out_ch, affine=True)
        # Second convolution layer
        self.conv1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 9), stride=1,
                                padding=(0, 4), bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch, affine=True)
        # Residual convolution layer
        self.conv_res = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 9), stride=2, 
                                padding=(0, 4), bias=True)
        self.bn_res = nn.BatchNorm2d(out_ch, affine=True)  
    def forward(self, x):

        out = self.conv0(x)
        out = self.bn0(out)
        out = F.relu(out)     

        out = self.conv1(out)
        out = self.bn1(out)  

        identity = self.conv_res(x)
        identity = self.bn_res(identity)
        identity = F.relu(identity)

        out += identity
        out = F.relu(out)

        return out


class TCResNet14(nn.Module):
    """ TC-ResNet14 implementation 

        Input dim:  [batch_size x n_mels x 1 x 101]
        Output dim: [batch_size x n_classes]

        Input -> Conv Layer -> (S2 Block -> S1 Block) x 3 -> Average pooling -> FC Layer -> Output
    """
    def __init__(self, args, k=1, n_mels=40, n_classes=10):
        super(TCResNet14, self).__init__()
        self.args = args
        # First Convolution layer
        self.conv_block = nn.Conv2d(in_channels=n_mels, out_channels=int(16*k), kernel_size=(1, 3), 
                                   padding = (0,1), bias=False)

        self.s2_block0 = S2_Block(int(16*k), int(24*k))
        self.s1_block0 = S1_Block(int(24*k))
        self.s2_block1 = S2_Block(int(24*k), int(32*k))
        self.s1_block1 = S1_Block(int(32*k))
        self.s2_block2 = S2_Block(int(32*k), int(48*k))
        self.s1_block2 = S1_Block(int(48*k))
        self.s1_block2_s = S1_Block(int(48*k))

        # Features are [batch x 48*k channels x 1 x 13] at this point
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 13), stride=1) 
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Conv2d(in_channels=int(48*k), out_channels=n_classes, kernel_size=1, padding=0, 
                            bias=False)

    def forward(self, x):
        out = self.conv_block(x)

        out = self.s2_block0(out)
        out = self.s1_block0(out)
        out = self.s2_block1(out)
        out = self.s1_block1(out)
        out = self.s2_block2(out)
        k_map = self.s1_block2(out)
        s_map = self.s1_block2_s(out)
        

        out = self.avg_pool(k_map)
        out = self.dropout(out)
        out = self.fc(out)
        out_k = out.view(out.shape[0], -1)
    
        return out_k, out_k, k_map, s_map

    def save(self, is_onnx=0):
        if (is_onnx):
            dummy_input = torch.randn(16, 40, 1, 101)
            torch.onnx.export(self, dummy_input, "TCResNet14.onnx", verbose=True, input_names=["input0"], output_names=["output0"])
        else:
            torch.save(self.state_dict(), "TCResNet14")

    def load(self):
        self.load_state_dict(torch.load("TCResNet14", map_location=lambda storage, loc: storage))


class TCResNet8(nn.Module):
    """ TC-ResNet8 using Depth-wise and Point-wise Convolution """

    def __init__(self,args,k=1, n_mels=40, n_classes=10, n_speaker=1861):
        super(TCResNet8, self).__init__()
        self.args = args
        # First Convolution layer (Depth-wise and Point-wise)
        self.dw_conv_block = nn.Conv2d(in_channels=n_mels, out_channels=int(16 * k), kernel_size=(1, 3),
                                       padding=(0, 1), bias=True)
        # self.pw_conv_block = nn.Conv2d(in_channels=n_mels, out_channels=int(16 * k), kernel_size=1, bias=False)

        # S2 Blocks
        self.s2_block0 = S2_Block(int(16 * k), int(24 * k))
        # self.s2_block0_speaker = S2_Block(int(16 * k), int(24 * k))

        self.s2_block1 = S2_Block(int(24 * k), int(32 * k))
        # self.s2_block1_speaker = S2_Block(int(24 * k), int(32 * k))

        self.s2_block2 = S2_Block(int(32 * k), int(48 * k))
        self.s2_block2_speaker = S2_Block(int(32 * k), int(48 * k))

        # Average Pooling and Fully Connected Layer
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 13), stride=1)
        self.fc = nn.Conv2d(in_channels=int(48 * k), out_channels=n_classes, kernel_size=1, padding=0, bias=True)
        # self.fc_s = nn.Conv2d(in_channels=int(48 * k), out_channels=n_speaker, kernel_size=1, padding=0, bias=True)

        # Parameters for orthogonal loss
        # self.orth_block = OrthBlock(feature_dim=48)
        # self.conv_ks = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(1, 3),
        #                                padding=(1, 1), bias=True)
        # self.conv_sk = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(1, 3),
        #                                padding=(1, 1), bias=True)
        # self.kws_attn = nn.MultiheadAttention(embed_dim=51, num_heads=1)
        # self.sid_attn = nn.MultiheadAttention(embed_dim=51, num_heads=1)
        self.attn_k_weights = None
        self.attn_s_weights = None


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

        k_map = self.avg_pool(k_map)
        s_map = self.avg_pool(s_map)
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
            k_map, s_map = self.orth_block(k_map, s_map)


        
        # print(k_map.shape)
        # kws after att
        # out_k = self.avg_pool(k_map)
        out_k = self.fc(k_map)
        out_k = F.softmax(out_k, dim=1)
        out_k = out_k.view(out_k.shape[0], -1)
        
        # speaker after att
        # out_s = self.avg_pool(s_map)
        # out_s = self.fc_s(s_map)
        # out_s = F.softmax(out_s, dim=1)
        # out_s = out_s.view(out_s.shape[0], -1)
    
        return out_k, s_map, k_map, s_map

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
    # TCResNet8 test
    model_tcresnet8 = TCResNet8(1, 40, 12)
    result_tcresnet8 = model_tcresnet8(x)
    print(result_tcresnet8)
    # TCResNet14 test
    # model_tcresnet14 = TCResNet14(1, 40, 12)
    # result_tcresnet14 = model_tcresnet14(x)