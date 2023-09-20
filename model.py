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




class S2_Block(nn.Module):
    """ S2 ConvBlock used in Temporal Convolutions
        -> CONV -> BN -> RELU -> CONV -> BN -> (+) -> RELU
        |_______-> CONV -> BN -> RELU ->________|
    """

    def __init__(self, in_ch, out_ch):
        super(S2_Block, self).__init__()

        # First convolution layer
        self.conv0 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 9), stride=2,
                               padding=(0, 4), bias=False)
        self.bn0 = nn.BatchNorm2d(out_ch, affine=True)
        # Second convolution layer
        self.conv1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 9), stride=1,
                               padding=(0, 4), bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, affine=True)
        # Residual convolution layer
        self.conv_res = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 9), stride=2,
                                  padding=(0, 4), bias=False)
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


class TCResNet8(nn.Module):
    """ TC-ResNet8 implementation.

        Input dim:  [batch_size x n_mels x 1 x 101]
        Output dim: [batch_size x n_classes]

        Input -> Conv Layer -> (S2 Block) x 3 -> Average pooling -> FC Layer -> Output
    """

    def __init__(self, k, n_mels, n_classes):
        super(TCResNet8, self).__init__()

        # First Convolution layer
        self.conv_block = nn.Conv2d(in_channels=n_mels, out_channels=int(16 * k), kernel_size=(1, 3),
                                    padding=(0, 1), bias=False)

        # S2 Blocks
        self.s2_block0 = S2_Block(int(16 * k), int(24 * k))
        self.s2_block1 = S2_Block(int(24 * k), int(32 * k))
        self.s2_block2 = S2_Block(int(32 * k), int(48 * k))

        # Features are [batch x 48*k channels x 1 x 13] at this point
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 13), stride=1)
        self.fc = nn.Conv2d(in_channels=int(48 * k), out_channels=n_classes, kernel_size=1, padding=0,
                            bias=False)

    def forward(self, x):
        # print("nn input shape: ",x.shape)
        out = self.conv_block(x)

        out = self.s2_block0(out)
        out = self.s2_block1(out)
        out = self.s2_block2(out)

        out = self.avg_pool(out)
        out = self.fc(out)
        out = F.softmax(out, dim=1)

        return out.view(out.shape[0], -1)

    def save(self, is_onnx=0, name="TCResNet8"):
        if (is_onnx):
            dummy_input = torch.randn(16, 40, 1, 101)
            torch.onnx.export(self, dummy_input, "TCResNet8.onnx", verbose=True, input_names=["input0"],
                              output_names=["output0"])
        else:
            torch.save(self.state_dict(), "saved_model/"+name)

    def load(self,name="TCResNet8"):
        self.load_state_dict(torch.load("saved_model/"+name, map_location=lambda storage, loc: storage))


class TCResNet8_flatten(nn.Module):
    """ TC-ResNet8 implementation.

        Input dim:  [batch_size x n_mels x 1 x 101]
        Output dim: [batch_size x n_classes]

        Input -> Conv Layer -> (S2 Block) x 3 -> pooling -> FC Layer -> Output
    """

    def __init__(self, k, n_mels, n_classes):
        super(TCResNet8_flatten, self).__init__()

        # First Convolution layer
        # self.conv_block = nn.Conv2d(in_channels=n_mels, out_channels=int(16 * k), kernel_size=(1, 3),
        #                             padding=(0, 1), bias=False)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=n_mels, out_channels=n_mels, kernel_size=(1, 3),
                      padding=(0, 1), groups=n_mels, bias=False),
            nn.Conv2d(in_channels=n_mels, out_channels=int(16 * k), kernel_size=(1, 1), bias=False)
        )

        # S2 Blocks
        # ###########################  self.s2_block0 = S2_Block(int(16 * k), int(24 * k))
        # self.conv0_0 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(1, 9), stride=2,
        #                        padding=(0, 4), groups=8, bias=False)
        self.conv_0_0 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3),
                      padding=(0, 1), groups=n_mels, bias=False),
            nn.Conv2d(in_channels=n_mels, out_channels=int(16 * k), kernel_size=(1, 1), bias=False)
        )

        self.bn0_0 = nn.BatchNorm2d(24, affine=True)
        self.conv0_1 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(1, 9), stride=1,
                               padding=(0, 4), bias=False)
        self.bn0_1 = nn.BatchNorm2d(24, affine=True)
        self.conv_res0 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(1, 9), stride=2,
                                  padding=(0, 4), bias=False)
        self.bn_res0 = nn.BatchNorm2d(24, affine=True)

        # ############################   self.s2_block1 = S2_Block(int(24 * k), int(32 * k))
        self.conv1_0 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(1, 9), stride=2,
                               padding=(0, 4), bias=False)
        self.bn1_0 = nn.BatchNorm2d(32, affine=True)
        self.conv1_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 9), stride=1,
                               padding=(0, 4), bias=False)
        self.bn1_1 = nn.BatchNorm2d(32, affine=True)
        self.conv_res1 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(1, 9), stride=2,
                                  padding=(0, 4), bias=False)
        self.bn_res1 = nn.BatchNorm2d(32, affine=True)

        # ############################   self.s2_block2 = S2_Block(int(32 * k), int(48 * k))
        self.conv2_0 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(1, 9), stride=2,
                               padding=(0, 4), bias=False)
        self.bn2_0 = nn.BatchNorm2d(48, affine=True)
        self.conv2_1 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(1, 9), stride=1,
                               padding=(0, 4), bias=False)
        self.bn2_1 = nn.BatchNorm2d(48, affine=True)
        self.conv_res2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(1, 9), stride=2,
                                  padding=(0, 4), bias=False)
        self.bn_res2 = nn.BatchNorm2d(48, affine=True)

        # Features are [batch x 48*k channels x 1 x 13] at this point
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 13), stride=1)
        self.fc = nn.Conv2d(in_channels=int(48 * k), out_channels=10, kernel_size=1, padding=0,
                            bias=False)




    def forward(self,x):
        # layer-wise training
        # out0 = self.forward_0(x)
        # out1 = self.forward_1(x)
        # out2 = self.forward_full(x)
        # return [out0,out1, out2]

        # end2end training
        x = F.pad(x, [0, 11])
        out = self.conv_block(x)

        ########## s1 - 0 ########
        x0 = out
        out = self.conv0_0(x0)
        out = self.bn0_0(out)
        out = F.relu(out)
        out = self.conv0_1(out)
        out = self.bn0_1(out)
        identity = self.conv_res0(x0)
        identity = self.bn_res0(identity)
        identity = F.relu(identity)
        out += identity
        out = F.relu(out)
        # out0 = out

        ########## s1 - 1 ########
        x1 = out
        out = self.conv1_0(x1)
        out = self.bn1_0(out)
        out = F.relu(out)
        out = self.conv1_1(out)
        out = self.bn1_1(out)
        identity = self.conv_res1(x1)
        identity = self.bn_res1(identity)
        identity = F.relu(identity)
        out += identity
        out = F.relu(out)
        # out1 = out
        ############### early - 1 ##########
        # out1 = F.avg_pool2d(identity, (1,identity.size()[3]))
        # out1 = self.early_fc_1(out1)
        # out1 = F.softmax(out1, dim=1)
        # out1 = out1.view(out0.shape[0], -1)

        ######### s1 - 2 ##########
        x2 = out
        out = self.conv2_0(x2)
        out = self.bn2_0(out)
        out = F.relu(out)
        out = self.conv2_1(out)
        out = self.bn2_1(out)
        identity = self.conv_res2(x2)
        identity = self.bn_res2(identity)
        identity = F.relu(identity)
        out += identity
        out = F.relu(out)
        # out2 = out
        #########################
        out2 = F.avg_pool2d(out, (1,out.size()[3]))
        out2 = self.fc(out2)
        out2 = F.softmax(out2, dim=1)
        out2 = out2.view(out2.shape[0], -1)

        return out2
        #ty
        #
        # out = self.conv0_0(out)
        # out = self.bn0_0(out)
        # out = self.conv0_1(out)
        # out = self.bn0_1(out)
        # out = self.conv_res0(out)
        # out = self.bn_res0(out)
        '''
        out0 = F.avg_pool2d(out,out.size()[3])
        out0 = self.early_fc_0(out0)
        out0 = F.softmax(out0, dim=1)
        out0 = out0.view(out0.shape[0], -1)

        out = self.s2_block1(out)
        out1 = F.avg_pool2d(out, kernel_size = (1,26))
        out1 = self.early_fc_1(out1)
        out1 = F.softmax(out1, dim=1)
        out1 = out1.view(out1.shape[0], -1)

        out = self.s2_block2(out)
        out = F.avg_pool2d(out, kernel_size=(1, 13))
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        out2 = out.view(out.shape[0], -1)
        '''



        # end2end training
        # out = self.conv_block(x)
        # out0  = F.avg_pool2d(out, kernel_size = (1,101))
        # out0 = self.early_fc_0(out0)
        # out0 = F.softmax(out0, dim=1)
        # out0 = out0.view(out0.shape[0], -1)
        #
        # out = self.s2_block0(out)
        # out1 = F.avg_pool2d(out, kernel_size=(1, 51))
        # out1 = self.early_fc_1(out1)
        # out1 = F.softmax(out1, dim=1)
        # out1 = out1.view(out1.shape[0], -1)
        #
        # out = self.s2_block1(out)
        # out = self.s2_block2(out)
        # out = F.avg_pool2d(out, kernel_size=(1, 13))
        # out = self.fc(out)
        # out = F.softmax(out, dim=1)
        # out2 = out.view(out.shape[0], -1)
        # return [out0, out1, out2]

    def forward_0(self, x):
        out = self.conv_block(x)
        out = self.s2_block0(out)
        out = F.avg_pool2d(out, kernel_size = (1,51))
        out = self.early_fc_0(out)
        out = F.softmax(out, dim=1)
        out = out.view(out.shape[0], -1)
        return out

    def forward_1(self, x):
        with torch.no_grad():
            out = self.conv_block(x)
            out = self.s2_block0(out)
        out = self.s2_block1(out)
        out = F.avg_pool2d(out, kernel_size=(1, 26))
        out = self.early_fc_1(out)
        out = F.softmax(out, dim=1)
        out = out.view(out.shape[0], -1)
        return out

    def forward_full(self, x):
        # print("nn input shape: ",x.shape)
        with torch.no_grad():
            out = self.conv_block(x)
            out = self.s2_block0(out)
            out = self.s2_block1(out)
        out = self.s2_block2(out)
        out = F.avg_pool2d(out, kernel_size=(1, 13))
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        out = out.view(out.shape[0], -1)
        return out




    def save(self, name, is_onnx=0):
        if (is_onnx):
            dummy_input = torch.randn(16, 40, 1, 101)
            torch.onnx.export(self, dummy_input, "TCResNet8.onnx", verbose=True, input_names=["input0"],output_names=["output0"])
        else:
            torch.save(self.state_dict(), name)

    def load(self,name="TCResNet8"):
        self.load_state_dict(torch.load("saved_model/"+name, map_location=lambda storage, loc: storage))





if __name__ == "__main__":
    x = torch.rand(1, 40, 1, 101)
    model_tcresnet8 = TCResNet8(1, 40, 12)
    result_tcresnet8 = model_tcresnet8(x)
    print(result_tcresnet8)
