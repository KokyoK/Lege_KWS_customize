import torch
torch.manual_seed(42)
import torch.utils.data as data
import speech_dataset as sd
# import utility as util
import utility_ee as util
import model as md
import model_quantize as mdq




TRAIN = True
# ROOT_DIR = "dataset/google_origin/"
# WORD_LIST = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
# ROOT_DIR = "dataset/lege/"
ROOT_DIR = "dataset/lege/"
WORD_LIST = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']


NUM_EPOCH = 500

if __name__ == "__main__":

    # model_fp32 = md.TCResNet8_flatten(k=1, n_mels=40, n_classes=len(WORD_LIST))
    model_fp32 = mdq.QuantizedTCResNet8(k=1, n_mels=40, n_classes=len(WORD_LIST))

    if TRAIN :

        util.train(model_fp32, ROOT_DIR, WORD_LIST, NUM_EPOCH)

    else:
        train, dev, test = sd.split_dataset(ROOT_DIR, WORD_LIST)
        ap = sd.AudioPreprocessor()
        test_data = sd.SpeechDataset(test, "eval", ap, WORD_LIST)
        test_dataloader = data.DataLoader(test_data, batch_size=1, shuffle=True)
        util.evaluate_testset(model_fp32, test_dataloader)
        



