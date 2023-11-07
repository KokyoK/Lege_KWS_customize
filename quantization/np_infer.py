import warnings
import sys
import pyaudio
import wave
import torch.nn as nn
import torch.fft
import speech_dataset as sd
import model_quantize as mdq

def conv(input_data, conv_kernel, bias, padding):

    conv_kernel = conv_kernel.detach().numpy()
    bias = bias.detach().numpy()
    input_data = input_data.detach().numpy()
    input_data = np.pad(input_data, padding, mode='constant')
    # 计算输出大小
    input_shape = input_data.shape
    kernel_shape = conv_kernel.shape
    output_shape = (input_shape[0], kernel_shape[0], input_shape[2], input_shape[3] - kernel_shape[3] + 1)

    # 初始化输出
    output_data = np.zeros(output_shape, dtype=np.float32)

    # 进行卷积操作
    for b in range(input_shape[0]):
        for c in range(kernel_shape[0]):
            for i in range(output_shape[2]):
                for j in range(output_shape[3]):
                    output_data[b, c, i, j] = np.sum(
                        # input_data[b, :, i:i + kernel_shape[2], j:j + kernel_shape[3]] * conv_kernel[c]) + bias[c]  # 加上偏置项
                        input_data[b, :, i:i + kernel_shape[2], j:j + kernel_shape[3]] * conv_kernel[c])  # 加上偏置项

    return output_data


def my_infer(feat):
    torch.set_printoptions(profile="full")
    s = model.state_dict()
    ### conv0

    conv1_w = s["conv_block_d.weight"]
    conv1_b = s["conv_block_d.bias"]
    # x =( feat/0.7466).round().clip(-128,127)
    x = (feat/s['quant_inp.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value']*128).round().clip(-128,127) # quant x
    int_conv1_w = (conv1_w / model.conv_block_d.quant_weight_scale() + model.conv_block_d.quant_weight_zero_point()).round()  # quant weight    在101维上进行padding
    int_conv1_b = (conv1_b /  model.conv_block_d.quant_bias_scale() + model.conv_block_d.quant_bias_zero_point()).round()
    out = F.conv2d(input=x, weight=int_conv1_w, bias= int_conv1_b,padding=(0, 1), stride=1)

    ################ conv d  ###########################
    x = x.squeeze()
    output = torch.zeros(16,101)
    int_conv1_w = int_conv1_w.squeeze()
    for oc in range(16):
        for ofeat in range(101):
            sum = 0
            for ic in range(40):
                for k in range(3):
                    idx = k + ofeat -1
                    if(idx>=0 and idx <101):
                        sum += x[ic, idx] * int_conv1_w[oc,ic,k]
            output[oc, ofeat] = sum + int_conv1_b[oc]
    print(output)
    ###################################################  int bias
    '''
    ################## conv reverse ##################
    # x = x.squeeze()
    # output = torch.zeros(16,101)
    # int_conv1_w = int_conv1_w.squeeze()
    # for ofeat in range(101):
    #     for k in range(3):
    #         for ic in range(40):
    #             for oc in range(16):
    #                 idx = k + ofeat -1
    #                 if(idx>=0 and idx <101):
    #                     output[oc, ofeat] += x[ic, idx] * int_conv1_w[oc,ic,k]
            # output[oc, ofeat] = sum
    # print(output)

    ######################################################
    '''
    m = s['quant_inp.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value'] *\
        model.conv_block_d.quant_weight_scale() / \
        s['relu0.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value']
    int_out = (torch.Tensor(out) * m * 2).round().clip(0,255)
    #
    # # s2_block_0 conv0 + Relu
    conv0_w_s0 = s["s2_block0.conv0_d.weight"]
    # input = int_out
    # int_conv0_s0_w = (conv0_w_s0 / model.s2_block0.conv0_d.quant_weight_scale() + model.s2_block0.conv0_d.quant_weight_zero_point()).round()
    # out = F.conv2d(input=input, weight=int_conv0_s0_w, padding=(0,4),stride=2)
    # m = s['relu0.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value'] * \
    #     model.s2_block0.conv0_d.quant_weight_scale() / \
    #     s['s2_block0.relu0.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value']
    # int_out = F.relu(torch.Tensor(out) * m).round().clip(0, 255)

    # print(int_out)
    # input = input.squeeze()
    # output = torch.zeros(24,51)
    # int_conv0_s0_w = int_conv0_s0_w.squeeze()
    # for oc in range(24):
    #     for ofeat in range(51):
    #         sum = 0
    #         for ic in range(16):
    #             for k in range(9):
    #                 idx = k + ofeat*2 -4
    #                 if(idx>=0 and idx <101):
    #                     sum += input[ic, idx] * int_conv0_s0_w[oc,ic,k]
    #         output[oc, ofeat] = sum
    # print(output)


    # s2_block_0 conv1 + ReLu
    # conv1_w_s0 = s["s2_block0.conv1_d.weight"]
    # input = int_out
    # int_conv1_s0_w = (conv1_w_s0 / model.s2_block0.conv1_d.quant_weight_scale() + model.s2_block0.conv1_d.quant_weight_zero_point()).round()                                            #                              padding=(0, 1), bias=False, groups=n_mels
    # out = F.conv2d(input=input, weight=int_conv1_s0_w, padding=(0,4),stride=1)
    # m = s['s2_block0.relu0.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value']* \
    #     model.s2_block0.conv1_d.quant_weight_scale() / \
    #     s['s2_block0.relu2.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value']
    # int_out = (torch.Tensor(out) * m).round().clip(0, 255)

    # maxpool
    # int_out =F.max_pool2d(int_out, kernel_size=(1, 51), stride=1)


    # fc layer + relu
    # input = int_out.reshape(int_out.shape[0],int_out.shape[1])
    # fc_w = s['fc.weight']
    # int_fc_w = (fc_w /model.fc.quant_weight_scale() + model.fc.quant_weight_zero_point()).round()
    # out = int_fc_w @ input.T
    # m = s['s2_block0.relu2.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value']* \
    #     model.fc.quant_weight_scale() / \
    #     s['relu2.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value']
    # int_out = (torch.Tensor(out) * m).round().clip(0, 255)
    # # print("manual_out",out)
    # return int_out

import numpy as np



nm = 17
def record_para(model):
    savedStdout = sys.stdout  # 保存标准输出流
    file =  open('input.txt', 'w+')
    sys.stdout = file  # 标准输出重定向至文件
    layers = [[model.conv_block_d.int_weight(),
              model.conv_block_d.int_bias()],
              # model.conv_block_p,
              [model.s2_block0.conv0_d.int_weight(),
              model.s2_block0.conv0_d.int_bias()],
              # model.s2_block0.conv0_p,
              [model.s2_block0.conv1_d.int_weight(),
              model.s2_block0.conv1_d.int_bias()],
              [model.fc0.int_weight(),None],
              # model.s2_block0.conv1_p,
              [model.s2_block1.conv0_d.int_weight(),
              model.s2_block1.conv0_d.int_bias()],
              # # model.s2_block1.conv0_p,
              [model.s2_block1.conv1_d.int_weight(),
              model.s2_block1.conv1_d.int_bias()],
              # # model.s2_block1.conv1_p,
              [model.s2_block2.conv0_d.int_weight(),
              model.s2_block2.conv0_d.int_bias()],
              # # model.s2_block2.conv0_p,
              [model.s2_block2.conv1_d.int_weight(),
              model.s2_block2.conv1_d.int_bias()],
              # # model.s2_block2.conv1_p,
              [model.fc.int_weight(),None]
              # model.fc.int_bias(),
              ]
    scales = [model.conv_block_d.quant_weight_scale(),
              model.s2_block0.conv0_d.quant_weight_scale(),
              model.s2_block0.conv1_d.quant_weight_scale(),
              model.fc0.quant_weight_scale(),
              model.s2_block1.conv0_d.quant_weight_scale(),
              model.s2_block1.conv1_d.quant_weight_scale(),
              model.s2_block2.conv0_d.quant_weight_scale(),
              model.s2_block2.conv1_d.quant_weight_scale(),
              model.fc.quant_weight_scale(),]
    s = model.state_dict()
    i = 0
    j = 0
    for key in s:
        if(key.endswith("value")):     # 打印 m的值
            if(i != 0): # activation的layer 不是第一层
                s1 = s[prev_key]
                s2 = scales[i-1]
                s3 = s[key]
                m0 = torch.tensor((s1*s2/s3 * 2 ** nm).round().int(),dtype=torch.int32)
                print("const MDT m0_{} = {};".format(i, int(m0)))
            # if not key.startswith("relu_e"):    # early classifier 不更新prev key，因为下一层的scale不从early 计算。需要注意early-fc的命名
            prev_key = key
            i += 1
        else:  # key.endswith("weight") or key.endswith("bias") 打印 weight bias
            # print(s[key])
            if key.endswith("weight"):
                type_str = "weight"
                type_idx = 0
            else:
                type_str = "bias"
                type_idx = 1
            block_name = key.replace(f".{type_str}","").replace(".","_")

            # scale = model.block_name.quant_weight_scale()
            # zp = model.block_name.quant_weight_zero_point()
            # int_w = (s[key] / scale +  zp).round().tolist()
            # int_w = s[key].int()
            para = layers[j][type_idx].squeeze()
            data_str = "{}".format(layers[j][0].squeeze().tolist()).replace("[","{").replace("]","}")
            if(len(para.shape) == 3):
                print(f"const {type_str[0].upper()}DT {type_str[0]}_{block_name}[{para.shape[0]}][{para.shape[1]}][{para.shape[2]}] = {data_str};")
            elif(len(para.shape) == 2):
                print(f"const {type_str[0].upper()}DT {type_str[0]}_{block_name}[{para.shape[0]}][{para.shape[1]}] = {data_str};")
            else:
                print(f"const {type_str[0].upper()}DT {type_str[0]}_{block_name}[{para.shape[0]}] = {data_str};")
            if type_str == "bias" or key.startswith("fc"):
                j += 1
        # break


if __name__ == "__main__":
    model = mdq.QuantizedTCResNet8(1, 40, 10)
    model.load("../saved_model/lege_ee_189_infer_94.667.pt")
    model.eval()

    '''
    # import brevitas.onnx as bo
    # export_onnx_path = "8b_weight_act_bias_net.onnx"
    # input_shape = (1, 40, 1, 101)
    # bo.export_finn_onnx(model, input_shape, export_onnx_path)
    '''
    # x = torch.randn(1, 40, 1, 101)
    # out = model(x)
    # record_para(model)
    # '''

    # x = torch.randn(1, 40, 1, 101)
    # out = model(x)
    #########  retain bias ###################
    for name, module in model.named_modules():
        # 检查模块是否是卷积层
        if isinstance(module, nn.Conv2d):
            # 打印卷积层的名字
            module.cache_inference_quant_bias = True
            # print(name)
    # model.conv_block_d.cache_inference_quant_bias = True
    # model.conv_block_d.cache_inference_quant_bias = True
    ######## end of retain bias ###################



    from speech_dataset import *
    root_dir = "../dataset/lege/"
    word_list = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
    loaders = sd.kws_loaders(root_dir, word_list, [])
    count_all = 0
    count_match = 0
    Bar = enumerate(loaders[0])

    for i, (train_spectrogram, train_labels) in Bar:        # NOTE batch size 必须设为1才能进 my_infer 函数
        out = model(train_spectrogram)
        record_para(model)
        # my_out = my_infer(train_spectrogram)
        # train_spectrogram, out = np.array(train_spectrogram),np.array(out.tensor.detach())
        # train_spectrogram = train_spectrogram.squeeze()
        # np.save('quantization/input.npy', train_spectrogram)


        #

        # out = torch.tensor(out.argmax(axis=1))
        # my_out = my_out.argmax(axis=0)
        # print("true out: ", out)
        # print("my   out: ", my_out)
        # if(out == my_out):
        #     count_match += 1
        # count_all +=1
        # print(train_spectrogram.shape, train_labels.shape)
        # print("")
        break
        print("")


