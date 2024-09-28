import os
import sys
current_file_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_file_path))  
sys.path.append(root_dir)
import torch
torch.manual_seed(41)
import torch.utils.data as data
import speech_dataset as sd
import noisy_dataset as nd
# import utility as util
import utility_ee as util
import model as md
import argparse
import os
from SNN import *
import util_snn

# sys.path.append(parent_dir)

parser = argparse.ArgumentParser(description='SNN')
parser.add_argument('--dataset', default="google",  help='google | lege')
# parser.add_argument('--device', default="board", help='')
parser.add_argument('--orth', default="yes", help='')
parser.add_argument('--denoise', default="yes", help='')



parser.add_argument('--log', default="logs/record.csv", help='')
parser.add_argument('--ptname', default="our", help='')
parser.add_argument('--train', default="yes", help='')
parser.add_argument('--denoise_loss', default="yes", help='')
parser.add_argument('--orth_loss', default="yes", help='')
parser.add_argument('--backbone', default="star", help='res | star ｜ bc | decouple | tc14 | kwt')
parser.add_argument('--denoise_net', default="specu", help='')
parser.add_argument('--feat', default="spec", help='spec | mfcc ')
parser.add_argument('--att', default="no", help='')


args = parser.parse_args()
loaders = torch.load(f"loaders/loaders_{args.dataset}_{args.feat}.pth")
print("Get loaders done.")
TRAIN = False
if args.train == "yes":
    TRAIN = True
# ROOT_DIR = "dataset/google_origin/"
if args.dataset == "google":
    if args.feat == "spec":
        ROOT_DIR = "dataset/google_noisy/NGSCD_SPEC/" 
    elif args.feat == "mfcc":
        ROOT_DIR = "dataset/google_noisy/NGSCD_MFCC/" 
    WORD_LIST = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
elif args.dataset == "lege":
    ROOT_DIR = "dataset_lege/lege_noisy/NGSCD_SPEC/"
    WORD_LIST = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
    
    SPEAKER_LIST = nd.fetch_speaker_list(ROOT_DIR, WORD_LIST)
NUM_EPOCH = 200

print("dataset root:", ROOT_DIR)
print("keyword number:", len(WORD_LIST))
# print("speaker number:", len(SPEAKER_LIST))
if __name__ == "__main__":
    model = SpikGRU()
    # model =  SNNKeywordSpotting()

    if TRAIN :
        # model_fp32.load("google/baseline_308_kwsacc_92.05_idloss_0.0394")
        # model_fp32.load("google_noisy/cammd_18_kwsacc_83.18_idloss_0.2399")
        # model_fp32.load("oh_73_kwsacc_84.94_idloss_0.2419") # oh
        # model_fp32.load("google_noisy/kwt_143_kwsacc_77.27_idloss_0.1912") # oh

        util_snn.train(model, NUM_EPOCH,loaders,args)