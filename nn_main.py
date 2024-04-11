import torch
torch.manual_seed(42)
import torch.utils.data as data
import speech_dataset as sd
# import utility as util
import utility_ee as util
import model as md
import argparse
import os



parser = argparse.ArgumentParser(description='Keyword Spotting')
# parser.add_argument('--sample_num', default=100, type=float, help='sample_num')
# parser.add_argument('--vf_count', default=8, type=int, help='vf_count')
# parser.add_argument('--model_name', default="mobilev2_3", help='model_name\noptions: "resnet32, "tcresnet8"')
# parser.add_argument('--dataset', default="cifar10", help='dataset_name\noptions: "cifar10, "google_kws"')
parser.add_argument('--k', default=1, type=float, help='')
parser.add_argument('--s', default=1, type=float, help='')

parser.add_argument('--log', default="logs/record.csv", help='')
parser.add_argument('--ptname', default="sim", help='')
parser.add_argument('--train', default="yes", help='')
args = parser.parse_args()

TRAIN = False
if args.train == "yes":
    TRAIN = True
ROOT_DIR = "dataset/google_origin_SPEC/"
WORD_LIST = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
# ROOT_DIR = "../EarlyExit/dataset/huawei_modify/WAV_new/"
# WORD_LIST = ['hey_celia', '支付宝扫一扫', '停止播放', '下一首', '播放音乐', '微信支付', '关闭降噪', '小艺小艺', '调小音量', '开启透传']

# # SPEAKER_LIST = [speaker for speaker in os.listdir("dataset/huawei_modify/WAV/") if speaker.startswith("A")]
# ROOT_DIR = "dataset/lege/"
# WORD_LIST = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
SPEAKER_LIST = sd.fetch_speaker_list(ROOT_DIR, WORD_LIST)
NUM_EPOCH = 1000

print("dataset root:", ROOT_DIR)
print("keyword number:", len(WORD_LIST))
print("speaker number:", len(SPEAKER_LIST))
if __name__ == "__main__":
    model_fp32 = md.SiameseTCResNet(k=1, n_mels=40, n_classes=len(WORD_LIST),n_speaker=len(SPEAKER_LIST))

    loaders = sd.get_loaders( ROOT_DIR, WORD_LIST,SPEAKER_LIST)

    if TRAIN :
        # model_fp32.load("google/sim_472_kwsacc_90.12_idloss_0.0316")
        util.train(model_fp32, NUM_EPOCH,loaders, args)

    else:
        # model_fp32.load("google_sim_att_165_kwsacc_91.22_idloss_0.0571")
        # model_fp32.load("google/sim_att_12_kwsacc_90.46_idloss_0.0421")
        util.evaluate_testset(model_fp32, loaders[2])
        



