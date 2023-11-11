import torch
torch.manual_seed(42)
import torch.utils.data as data
import speech_dataset as sd
# import utility as util
import utility_ee as util
import model as md

import os


TRAIN = True
# ROOT_DIR = "dataset/google_origin/"
# WORD_LIST = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
# ROOT_DIR = "../EarlyExit/dataset/huawei_modify/WAV_new/"
# WORD_LIST = ['hey_celia', '支付宝扫一扫', '停止播放', '下一首', '播放音乐', '微信支付', '关闭降噪', '小艺小艺', '调小音量', '开启透传']

# # SPEAKER_LIST = [speaker for speaker in os.listdir("dataset/huawei_modify/WAV/") if speaker.startswith("A")]
ROOT_DIR = "dataset/lege/"
WORD_LIST = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
SPEAKER_LIST = sd.fetch_speaker_list(ROOT_DIR, WORD_LIST)
NUM_EPOCH = 5000

print("dataset root:", ROOT_DIR)
print("keyword number:", len(WORD_LIST))
print("speaker number:", len(SPEAKER_LIST))
if __name__ == "__main__":
    model_fp32 = md.TCResNet8(k=1, n_mels=40, n_classes=len(WORD_LIST),n_speaker=len(SPEAKER_LIST))
    if TRAIN :
        util.train(model_fp32, ROOT_DIR, WORD_LIST,SPEAKER_LIST, NUM_EPOCH)

    else:
        train, dev, test = sd.split_dataset(ROOT_DIR, WORD_LIST, SPEAKER_LIST)
        ap = sd.AudioPreprocessor()
        test_data = sd.SpeechDataset(test, "eval", ap, WORD_LIST)
        test_dataloader = data.DataLoader(test_data, batch_size=1, shuffle=True)
        util.evaluate_testset(model_fp32, test_dataloader)
        



