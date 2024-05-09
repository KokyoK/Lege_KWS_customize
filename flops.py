import torch
import torch.nn as nn
import torch.nn.functional as F
# import speech_dataset as sd
# import noisy_dataset as nsd 
# from timm.models.layers import DropPath, trunc_normal_
# from timm.models.registry import register_model
import thop
import time
# from ptflops import get_model_complexity_info
import copy
from argparse import Namespace

from unet import UNet,StarNet
from models.BCResNet import BCResNet
from model import TCResNet8
from models.SpecUNet import SpecUNet
from models.DecoupleNet import DecoupleNet
# from torch_flops import *


if __name__ == "__main__":
    args=Namespace()
    args.orth_loss = "no"
    args.denoise = "no"
    device = "cpu"
    
    
    input_tensor = torch.randn(1, 40, 1, 101).to(device)  # batch_size=4
    clean_tensor = torch.randn(1, 40, 1, 101).to(device) 

    # output_tensor = starnet(input_tensor)
    # print(output_tensor.shape)

    unet = UNet().to(device)
    flops, params = thop.profile(unet,inputs=(input_tensor,))
    print(f"U NET flops {flops}, params {params}")
    
    # flops_counter = TorchFLOPsByFX(unet)
    # flops_counter.propagate(input_tensor)
    # flops_counter.print_result_table()
    # print("-----------------------------------")
    specu = SpecUNet().to(device)
    flops, params = thop.profile(specu,inputs=(input_tensor,))
    print(f"spec NET flops {flops}, params {params}")
    

    print("-----------------------------------")
    starnet = StarNet(args=args)
    macs, params = thop.profile(starnet,inputs=(input_tensor,))
    print(f"STAR NET macs {macs}, params {params}")

    tcresnet = TCResNet8(args=args)
    macs, params = thop.profile(tcresnet,inputs=(input_tensor,))
    print(f"TC NET macs {macs}, params {params}")

    bcresnet = BCResNet(args=args)
    macs, params = thop.profile(bcresnet,inputs=(input_tensor,))
    print(f"BC NET macs {macs}, params {params}")

    dnet = DecoupleNet(args=args)
    macs, params = thop.profile(dnet,inputs=(input_tensor,))
    print(f"Decouple NET macs {macs}, params {params}")


        # start = time.time()
    # for i in range(1000):
    #     out = starnet(input_tensor)
    # end = time.time()
    # print("Running time of start net: ", (end-start)) # ms