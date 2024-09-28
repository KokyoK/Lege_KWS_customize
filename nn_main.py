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


parser = argparse.ArgumentParser(description='Keyword spotting')
# parser.add_argument('--a', default=0.5, type=float,help='weight of exit samples')
# parser.add_argument('--dataset', default="lege", help='dataset name, options: "cifar10", "google_kws","lege_kws"')
# parser.add_argument('--opt_method', default="", help='optimization method\noptions: "heuristic", "ippp", "sa",\ndefault no method')
# parser.add_argument('--latency_constraint', default=0.4, type=float, help='latency constraint')
# parser.add_argument('--sample_num', default=100, type=float, help='sample_num')
# parser.add_argument('--vf_count', default=8, type=int, help='vf_count')
# parser.add_argument('--model_name', default="tcresnet8_2", help='model_name\noptions: "mobilev2_2 , tcresnet8_2"')
# parser.add_argument('--dataset', default="google_kws", help='dataset_name\noptions: "cifar10, "google_kws"')

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
print(args)
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
# ROOT_DIR = "../EarlyExit/dataset/huawei_modify/WAV_new/"
# WORD_LIST = ['hey_celia', '支付宝扫一扫', '停止播放', '下一首', '播放音乐', '微信支付', '关闭降噪', '小艺小艺', '调小音量', '开启透传']

# # SPEAKER_LIST = [speaker for speaker in os.listdir("dataset/huawei_modify/WAV/") if speaker.startswith("A")]

SPEAKER_LIST = nd.fetch_speaker_list(ROOT_DIR, WORD_LIST)
NUM_EPOCH = 200

print("dataset root:", ROOT_DIR)
print("keyword number:", len(WORD_LIST))
print("speaker number:", len(SPEAKER_LIST))
if __name__ == "__main__":
    model_fp32 = md.SiameseTCResNet(k=1, n_mels=40, n_classes=len(WORD_LIST),n_speaker=len(SPEAKER_LIST),args=args)
    print("Get models done.")
    # loaders = sd.get_loaders( ROOT_DIR, WORD_LIST,SPEAKER_LIST)
    # loaders = nd.get_loaders( ROOT_DIR, WORD_LIST,SPEAKER_LIST,args)
    # torch.save(loaders,"loaders/loaders_google_noi.pth")
    # loaders = torch.load(f"loaders/loaders_{args.dataset}_noi.pth")
    # loaders = torch.load(f"loaders/loaders_{args.dataset}_align.pth")
    loaders = torch.load(f"loaders/loaders_{args.dataset}_{args.feat}.pth")
    print("Get loaders done.")
    model_fp32.set_args(args)

    if TRAIN :
        # model_fp32.load("google/baseline_308_kwsacc_92.05_idloss_0.0394")
        # model_fp32.load("google_noisy/cammd_18_kwsacc_83.18_idloss_0.2399")
        # model_fp32.load("oh_73_kwsacc_84.94_idloss_0.2419") # oh
        model_fp32.load("google_noisy/kwt_143_kwsacc_77.27_idloss_0.1912") # oh

        util.train(model_fp32, NUM_EPOCH,loaders,args)

    else:

        
        
        # # my
        # # python nn_main.py  --train no 
        # model_fp32.load("cammd_3_kwsacc_84.03_idloss_0.2370")
        
        # # bc
        # # python nn_main.py --train no --denoise_loss no --orth_loss no --att no --backbone bc --denoise_net specu --ptname tc
        # model_fp32.load("tc_67_kwsacc_80.82_idloss_0.5152")
        # model_fp32.load("bc_base_199_kwsacc_82.86_idloss_0.4171")
        
        # # tc
        # # python nn_main.py --train no --denoise_loss no --orth_loss no --att no --backbone tc --denoise_net specu --ptname tc
        # model_fp32.load("tc_67_kwsacc_80.82_idloss_0.5152")
        
        # star
        # python nn_main.py --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname star --train no
        model_fp32.load("oht_20_kwsacc_85.46_idloss_0.2305")
        
        # decouple
        # python nn_main.py --denoise_loss no --orth_loss no --att no --backbone decouple --denoise_net specu --ptname star --train no --feat mfcc
        # model_fp32.load("google_noisy/dmfccx_145_kwsacc_84.98_idloss_0.1745")
        
        # kwt 
        # python nn_main.py --denoise_loss no --orth_loss no --att no --backbone kwt --denoise_net unet --ptname star --train no --feat mfcc
        # model_fp32.load("kwt2_17_kwsacc_84.02_idloss_0.1642")
        
        # python nn_main.py  --train no --denoise_loss no --orth_loss yes --att no --backbone star --denoise_net specu --ptname star 
        # model_fp32.load("LSN+MM_168_kwsacc_84.31_idloss_0.1821")
        
        # python nn_main.py  --train no --denoise_loss yes --orth_loss no --att no --backbone star --denoise_net specu --ptname star 
        # model_fp32.load("LSN+SUB_1_kwsacc_85.88_idloss_0.1635")
        
        # python nn_main.py  --train no --denoise_loss no --orth_loss no --att no --backbone star --denoise_net specu --ptname star 
        # model_fp32.load("LSN_56_kwsacc_83.27_idloss_0.1687")
        
        
        
        # util.evaluate_testset_all(model_fp32, loaders[2],args)
        util.evaluate_testset_denoise(model_fp32, loaders[2],args)
        # util.evaluate_testset_all(model_fp32, loaders[2],args)
        



