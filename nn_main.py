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

# parser.add_argument('--device', default="board", help='')
parser.add_argument('--orth', default="yes", help='')
parser.add_argument('--denoise', default="yes", help='')
args = parser.parse_args()
print(args)


TRAIN = True
# ROOT_DIR = "dataset/google_origin/"
ROOT_DIR = "dataset/google_noisy/NGSCD_SPEC/"
WORD_LIST = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
# ROOT_DIR = "../EarlyExit/dataset/huawei_modify/WAV_new/"
# WORD_LIST = ['hey_celia', '支付宝扫一扫', '停止播放', '下一首', '播放音乐', '微信支付', '关闭降噪', '小艺小艺', '调小音量', '开启透传']

# # SPEAKER_LIST = [speaker for speaker in os.listdir("dataset/huawei_modify/WAV/") if speaker.startswith("A")]
# ROOT_DIR = "dataset/lege/"
# WORD_LIST = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
SPEAKER_LIST = nd.fetch_speaker_list(ROOT_DIR, WORD_LIST)
NUM_EPOCH = 200

print("dataset root:", ROOT_DIR)
print("keyword number:", len(WORD_LIST))
print("speaker number:", len(SPEAKER_LIST))
if __name__ == "__main__":
    model_fp32 = md.SiameseTCResNet(k=1, n_mels=40, n_classes=len(WORD_LIST),n_speaker=len(SPEAKER_LIST))

    # loaders = sd.get_loaders( ROOT_DIR, WORD_LIST,SPEAKER_LIST)
    loaders = sd.get_loaders( ROOT_DIR, WORD_LIST,SPEAKER_LIST)
    print("Get loaders done.")
    model_fp32.set_args(args)

    if TRAIN :
        # model_fp32.load("google/baseline_308_kwsacc_92.05_idloss_0.0394")
        util.train(model_fp32, NUM_EPOCH,loaders,args)

    else:
        # model_fp32.load("google_sim_att_165_kwsacc_91.22_idloss_0.0571")
        model_fp32.load("google_noisy/no_denoiseorth_45_kwsacc_85.49_idloss_0.5300_eer_0.4405")
        util.evaluate_testset(model_fp32, loaders[2],args)
        



