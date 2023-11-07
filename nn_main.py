import torch
torch.manual_seed(42)
import torch.utils.data as data
import speech_dataset as sd
# import utility as util
import utility as util
import model as md
import model_quantize as mdq
import argparse

parser = argparse.ArgumentParser(description='Early-exit aware Training')
parser.add_argument('--a', default=0.5, type=float,help='weight of exit samples')
parser.add_argument('--dataset', default="lege", help='dataset name, options: "cifar10", "google_kws","lege_kws"')
parser.add_argument('--opt_method', default="", help='optimization method\noptions: "heuristic", "ippp", "sa",\ndefault no method')
parser.add_argument('--latency_constraint', default=0.7, type=float, help='latency constranit')
parser.add_argument('--model_name', default="resnet32", help='model_name\noptions: "resnet32, "tcresnet8"')
args = parser.parse_args()



TRAIN = False
# ROOT_DIR = "../KWS_TCResNet/dataset/google_origin/"
# WORD_LIST = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
ROOT_DIR = "dataset/lege/"
WORD_LIST = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
SPEAKER_LIST=[]


NUM_EPOCH = 500

if __name__ == "__main__":

    # model_fp32 = md.TCResNet8_flatten(k=1, n_mels=40, n_classes=len(WORD_LIST))
    model_fp32 = mdq.QuantizedTCResNet8(k=1, n_mels=40, n_classes=len(WORD_LIST))
    loaders = sd.kws_loaders(ROOT_DIR, WORD_LIST,SPEAKER_LIST)
    # [train_loader, eval_loader, test_loader] = loaders
    if TRAIN :
        util.train(model_fp32, loaders, NUM_EPOCH,args)
    else:
        util.evaluate_ee_testset(model_fp32, loaders[2],thresholds=[0.9,0],name="saved_model/lege_ee_189_infer_94.667.pt")
        



