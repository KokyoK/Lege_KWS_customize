import os
from time import sleep
import torchaudio.functional as F_audio
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import model as md
import speech_dataset as sd
import torch.nn.functional as F
import argparse
import utility_ee as util
import pyaudio
from collections import deque

torch.manual_seed(42)
train_on_gpu = torch.cuda.is_available()

WORD_LIST = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
root_dir = "dataset/lege/"
SPEAKER_LIST = sd.fetch_speaker_list(root_dir, WORD_LIST)
sim_model = md.SiameseTCResNet(k=1, n_mels=40, n_classes=len(WORD_LIST), n_speaker=len(SPEAKER_LIST))
sim_model.load("sim_244_kwsacc_92.08_idloss_0.0728")
model = sim_model.network
model.eval()
anchors = []

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 16000
RECORD_SECONDS = 2
BUFFER_LENGTH = RATE * RECORD_SECONDS

# Shared deque for the audio buffer
audio_buffer = deque(maxlen=BUFFER_LENGTH // CHUNK)

# 新增变量来选择数据来源
source_type = "file"  # 可以是 "stream" 或 "file"
# source_type = "stream"
# 从文件读取时使用的文件路径
anchors_file = [
    root_dir + "站/A010_002.wav",
    root_dir + "升高/A010_003.wav",
    root_dir + "复位/A010_002.wav",
]
src_file = root_dir + "站/A010_002.wav"


def preprocess(tensor):
    # 重采样音频数据至8000 Hz
    tensor = F_audio.resample(tensor, 16000, 8000)

    # 创建梅尔频谱图变换
    spectrogram = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,  # 使用重采样后的采样率
            n_fft=480,  # 窗口长度为 30ms
            hop_length=160,  # 步长为 10ms
            f_min=0,
            f_max=4000,  # 由于重采样，最大频率为4000 Hz
            n_mels=40,
            window_fn=torch.hann_window
        ),
        torchaudio.transforms.AmplitudeToDB()
    )

    # 确保音频长度为2秒
    data_len = 16000  # 2秒音频长度 @ 8000 Hz
    tensor = tensor.squeeze()

    # 对于长度不足2秒的音频进行填充
    if tensor.shape[0] < data_len:
        tensor = F.pad(tensor, pad=(0, data_len - tensor.shape[0]), mode='constant', value=0)

    # 对于长度超过2秒的音频进行裁剪
    if tensor.shape[0] > data_len:
        t = tensor.shape[0] - data_len
        tensor = tensor[t:data_len + t]

    # 应用频谱图变换
    # print(tensor.shape)
    audio_data = spectrogram(tensor.view(1, -1)).view(1,40,1,101)
    # print(audio_data.shape)

    return audio_data


def infer(tensor):
    if source_type == "stream":
        # 如果是实时流，则检查音频强度，以判断是否为静音
        audio_strength = torch.mean(torch.abs(tensor.type(torch.float)))
        if audio_strength < torch.tensor(500):
            print("Silence detected")
            return

    # elif source_type == "file":
    #     # 如果是文件，则从指定文件路径加载音频
    #     tensor = torchaudio.load(src_file)[0]

    # 预处理音频数据
    audio_data = preprocess(tensor)

    # 使用模型进行推理
    out_kw, out_id, map_kw, map_s = model(x=audio_data)

    # 输出关键词识别结果
    kw_prob = out_kw.max(dim=1).values.item()
    print(f"Detected Keyword: {WORD_LIST[out_kw.argmax(dim=1)]} with probability {kw_prob:.2f}")

    # 对于每个已注册的发言者，计算与当前音频的相似度
    for name, anchor in anchors:
        distance = torch.dist(anchor,map_s, p=2)
        if distance < 20:
            print(f"Speaker {name} distance: {distance:.5f}")
        else:
            print("Unrecognized Speaker")

    print("++++++++++++++++++++++++++++++++++++")


def record_and_register():
    print("+++++++++++++ Register ++++++++++++++++")
    user_name = input("Enter your name: ")
    if not user_name:
        print("Registration cancelled.")
        return

    SPEAKER_LIST.append(user_name)
    sub_anchors = []
    for i in range(3):
        print(f"Recording for {user_name}, sample {i+1}/3...")
        if source_type == "stream":
            audio_data = record_audio()
            tensor = torch.tensor(np.frombuffer(audio_data, dtype=np.int16), dtype=torch.float32)
        elif source_type == "file":
            tensor = torchaudio.load(anchors_file[i])[0]

        audio_data = preprocess(tensor)
        out_kw, out_id, map_kw, map_s = model(x=audio_data)
        sub_anchors.append(map_s)
        infer(tensor)
    anchor = sum(sub_anchors) / len(sub_anchors)
    anchors.append([user_name, anchor])
    print(f"+++++++++++++ {user_name} Registered Successfully ++++++++++++++++")


def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return b''.join(frames)

def continuous_recording_and_inference():
    if source_type =="stream":
        while True:
            print("\nRecording...")
            audio_data = record_audio()
            tensor = torch.tensor(np.frombuffer(audio_data, dtype=np.int16), dtype=torch.float32)
            infer(tensor)
            sleep(2)
    elif source_type == "file":
        input_file = input("Type file name: ")
        input_file = os.path.join(root_dir,input_file)
        tensor = torchaudio.load( input_file)[0]
        # audio_data = preprocess(tensor)
        infer(tensor)

if __name__ == "__main__":
    while True:
        print("\n1: Register")
        print("2: Start Test")
        print("3: Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            record_and_register()
        elif choice == "2":
            continuous_recording_and_inference()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")
