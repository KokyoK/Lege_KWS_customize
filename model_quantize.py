
import torch
from brevitas.quant_tensor import QuantTensor

torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int32Bias


###############################  parameters #######################
qw = 8
qa = 8



class Quantized_S2_Block(nn.Module):
    """ Quantized S2 ConvBlock used in Temporal Convolutions
        - DCONV: depth-wise conv
        - PCONV point-wise conv
        -> DCONV -> RELU -> PCONV -> RELU -> DCONV -> RELU -> PCONV -> RELU
    """

    def __init__(self, in_ch, out_ch):

        super(Quantized_S2_Block, self).__init__()
        # self.conv0_d = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=(1, 9), stride=2, padding=(0, 4),
        #                          bias=False, groups=in_ch)
        self.conv0_d = qnn.QuantConv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=(1, 9), stride=2,
                                       padding=(0, 4), bias=True, weight_bit_width=qw, bias_quant=Int32Bias)
        self.relu0 = qnn.QuantReLU(bit_width=qw, return_quant_tensor=True)

        # self.conv0_p = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 1), stride=1, bias=False)
        self.conv0_p = qnn.QuantConv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 1), stride=1,
                                       bias=False, weight_bit_width=qw, bias_quant=Int32Bias)
        self.relu1 = qnn.QuantReLU(bit_width=qw, return_quant_tensor=True)

        # self.conv1_d = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 9), stride=1, padding=(0, 4),
        #                          bias=False, groups=out_ch)
        self.conv1_d = qnn.QuantConv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 9), stride=1,
                                       padding=(0, 4), bias=True, weight_bit_width=qw, bias_quant=Int32Bias)
        self.relu2 = qnn.QuantReLU(bit_width=qw, return_quant_tensor=True)

        # self.conv1_p = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 1), stride=1, bias=False)
        self.conv1_p = qnn.QuantConv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 1), stride=1,
                                       bias=False, weight_bit_width=qw, bias_quant=Int32Bias)
        self.relu3 = qnn.QuantReLU(bit_width=qw, return_quant_tensor=True)

        # self.conv_res_d = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=(1, 9), stride=2, padding=(0, 4),
        #                             bias=False, groups=in_ch)
        # self.conv_res_d = qnn.QuantConv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 9), stride=2,
        #                                padding=(0, 4), bias=False, weight_bit_width=qw, bias_quant=Int32Bias)
        # self.relu4 = qnn.QuantReLU(bit_width=qw, return_quant_tensor=True)

        # self.conv_res_p = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 1), stride=1, bias=False)
        # self.conv_res_p = qnn.QuantConv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 1), stride=1,
        #                                   bias=False, weight_bit_width=qw, bias_quant=Int32Bias)
        # self.relu5 = qnn.QuantReLU(bit_width=qw, return_quant_tensor=True)
        # self.quant_inp = qnn.QuantIdentity(bit_width=qa, return_quant_tensor=True)



    def forward(self, x):
        out = self.conv0_d(x)
        out = self.relu0(out)
        out = self.conv0_p(out)
        out = self.relu1(out)

        out = self.conv1_d(out)
        out = self.relu2(out)
        out = self.conv1_p(out)
        out = self.relu3(out)
        # out = self.bn1(out)
        # xi = self.conv_res_d(x)
        # xi = self.relu2(xi)
        # identity = xi

        # identity = self.conv_res_p(xi)
        # # identity = self.bn_res(identity)
        # identity = self.relu5(identity)
        # # out = out.skip_add(identity)

        # out = self.quant_inp(out)
        # identity = self.quant_inp(identity)
        # out = QuantTensor.add(out,identity)

        # out = out + identity
        # out = self.quant_inp(out)
        return out



class QuantizedTCResNet8(nn.Module):
    """ TC-ResNet8 implementation.

        Input dim:  [batch_size x n_mels x 1 x 101]
        Output dim: [batch_size x n_classes]

        Input -> Conv Layer -> (S2 Block) x 3 -> Average pooling -> FC Layer -> Output
    """

    def __init__(self, k, n_mels, n_classes):
        super(QuantizedTCResNet8, self).__init__()
        self.mode = "train"
        self.quant_inp = qnn.QuantIdentity(bit_width=qa, return_quant_tensor=True)

        # self.conv_block_d = nn.Conv2d(in_channels=n_mels, out_channels=n_mels, kernel_size=(1, 3), stride=1,
        #                               padding=(0, 1), bias=False, groups=n_mels)

        # self.conv_block_d = qnn.QuantConv2d(in_channels=n_mels, out_channels=n_mels, kernel_size=(1, 3), stride=1,
        #                                     bias=True, groups=n_mels, weight_bit_width=qw, padding=(0, 1), bias_quant=Int32Bias,
        #                                     )
        self.conv_block_d = qnn.QuantConv2d(in_channels=n_mels, out_channels=n_mels, kernel_size=(1, 3), stride=1,
                                            bias=True,  weight_bit_width=qw, padding=(0, 1), bias_quant=Int32Bias)
        self.relu0 = qnn.QuantReLU(bit_width=qw,return_quant_tensor=True)
        # self.conv_block_p = nn.Conv2d(in_channels=n_mels, out_channels=int(16 * k), kernel_size=(1, 1), stride=1,
        #                               bias=False)
        self.conv_block_p = qnn.QuantConv2d(in_channels=n_mels, out_channels=int(16 * k), kernel_size=(1, 1), stride=1,
                                            bias=False, weight_bit_width=qw, bias_quant=Int32Bias)
        self.relu1 = qnn.QuantReLU(bit_width=qw, return_quant_tensor=True)

        # S2 Blocks
        self.s2_block0 = Quantized_S2_Block(int(16 * k), int(24 * k))
        self.s2_block1 = Quantized_S2_Block(int(24 * k), int(32 * k))
        self.s2_block2 = Quantized_S2_Block(int(32 * k), int(48 * k))


        # Features are [batch x 48*k channels x 1 x 13] at this point
        # self.avg_pool = qnn.QuantAvgPool2d(kernel_size=(1, 13), stride=1)

        # self.avg_pool = nn.AvgPool2d(kernel_size=(1, 13), stride=1)

        # self.dropout = nn.Dropout(p=0.5)

        # self.fc = nn.Conv2d(in_channels=int(48 * k), out_channels=n_classes, kernel_size=1, padding=0,
        #                     bias=False)
        self.fc = qnn.QuantLinear(int(48 * k), n_classes, bias=False, weight_bit_width=qw)
        # self.fc = qnn.QuantLinear(int(24 * k), n_classes, bias=False, weight_bit_width=qw)
        self.relu2 = qnn.QuantReLU(bit_width=qw, return_quant_tensor=True)
        # self.fc = qnn.QuantConv2d(in_channels=int(48 * k), out_channels=n_classes, kernel_size=1, padding=0,
        #                                     bias=True, weight_bit_width=qw, bias_quant=Int32Bias)


    def forward(self, x):
        out = self.forward_path_train(x)
        return out

    def forward_path_train(self, x):
        x = self.quant_inp(x)
        out = self.conv_block_d(x)
        out = self.relu0(out)
        out = self.conv_block_p(out)
        out = self.relu1(out)

        out = self.s2_block0(out)
        out = self.s2_block1(out)
        out = self.s2_block2(out)
        # out = F.max_pool2d(out, kernel_size=(1, 51), stride=1)
        out = F.max_pool2d(out, kernel_size=(1, 13), stride=1)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        out = self.relu2(out)
        # out = F.softmax(out, dim=1)
        return out

    def save(self, is_onnx=0, name="saved_model/TCResNet8"):
        if (is_onnx):
            dummy_input = torch.randn(16, 30, 1, 101)
            torch.onnx.export(self, dummy_input, "TCResNet8.onnx", verbose=True, input_names=["input0"],
                              output_names=["output0"])
        else:
            torch.save(self.state_dict(), name)

            # f_save = open('activation.pkl', 'wb')
            # pickle.dump(min_max_act, f_save)
            # f_save.close()

    def load(self, name="saved_model/TCResNet8"):
        self.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))




if __name__ == "__main__":
    #################### for debug and deployment #########
    model = QuantizedTCResNet8(1, 40, 10)
    model.load("saved_model/TCResNet8")
    model.eval()

    from speech_dataset import *
    root_dir = "dataset/lege/"
    word_list = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
    # x = torch.rand(1, 40, 1, 101)
    # TCResNet8 test



    model.mode = "eval"
    model.eval()
    ap = AudioPreprocessor()
    train, val ,test= split_dataset(root_dir, word_list)

    # Dataset
    train_data = SpeechDataset(train, "train", ap, word_list)
    # Dataloaders
    train_dataloader = data.DataLoader(train_data, batch_size=1, shuffle=False)
    count_all = 0
    count_match = 0
    Bar = enumerate(train_dataloader)
    for i, (train_spectrogram, train_labels) in Bar:
        out = model(train_spectrogram)
        train_spectrogram, out = np.array(train_spectrogram),np.array(out.tensor.detach())
        # np.save('input.npy', train_spectrogram)
        # np.save('expected_output.npy', out)

        # my_out = my_infer(train_spectrogram)
        out = torch.tensor(out.argmax(axis=1))
        # my_out = my_out.argmax(axis=0)
        print("true out: ", out)
        # print("my   out: ", my_out)
        # if(out == my_out):
        #     count_match += 1
        # count_all +=1
        # print(train_spectrogram.shape, train_labels.shape)
        # print("")
        # break
    print("")


