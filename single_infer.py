import os
from time import sleep
from tkinter import simpledialog, messagebox
import torchaudio.functional as F_audio
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import torchaudio
import model as md
import speech_dataset as sd
import torch.nn.functional as F
import numpy as np
import argparse
import utility_ee as util

torch.manual_seed(42)
train_on_gpu = torch.cuda.is_available()
import tkinter as tk
import pyaudio
import torch
import threading
import queue
import numpy as np
from collections import deque


WORD_LIST = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
root_dir = "dataset/lege/"
SPEAKER_LIST = sd.fetch_speaker_list(root_dir,WORD_LIST)
sim_model = md.SiameseTCResNet(k=1, n_mels=40, n_classes=len(WORD_LIST), n_speaker=len(SPEAKER_LIST))
sim_model.load("sime_198_kw_95.149_simloss_0.008_.pt")
model = sim_model.network
# model.load("e_55_kw_91.683_valloss_43.267_.pt")
model.eval()
anchors =[]

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

######## for audio   # 从文件读取时使用的文件路径
anchors_file=[root_dir+"上升/A002_002.wav",
              root_dir+"小乐/A002_002.wav",
              root_dir+"降低/A002_002.wav",
              ]
# src_file = root_dir+"站/A002_002.wav"
src_file = root_dir+"升高/A005_002.wav"



def preprocess(tensor):
    tensor = F_audio.resample(tensor, 16000, 8000)
    spectrogram = nn.Sequential(
            # torchaudio.transforms.Resample(16000, 8000),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=480,          # window length = 30,
                hop_length=160,     # stride = 10
                f_min=0,
                f_max=8000,
                n_mels=40,
                window_fn=torch.hann_window
            ),
            torchaudio.transforms.AmplitudeToDB()
        )
    data_len = 16000
    tensor = tensor.squeeze()
    # Pad smaller audio files with zeros to reach 1 second (16_000 samples)
    if (tensor.shape[0] < data_len):
        tensor = F.pad(tensor, pad=(0, (data_len - tensor.shape[0])), mode='constant', value=0)

    # Clip larger audio files with zeros to reach 1 second (16_000 samples)
    if (tensor.shape[0] > data_len):
        t = tensor.shape[0] - data_len
        tensor = tensor[t:data_len+t]
    audio_data = spectrogram(tensor.view(1,-1)).view(1,40,1,101)

    return audio_data


 # Test example
gap = nn.AdaptiveAvgPool2d((1, 1))
# def register(tensor):
#     print("+++++++++++++ Register ++++++++++++++++")
#     user_name = simpledialog.askstring("Input", "Enter your name:", parent=root)
#     if user_name:
#         SPEAKER_LIST.append(user_name)
#         audio_data = preprocess(tensor)
#         out_kw, out_id, map_kw, map_s = model(x=audio_data)
#         anchor = map_s
#         # anchor = F.avg_pool2d(anchor, (1, 13))
#         anchor = gap(anchor)
#         anchors.append([user_name,anchor])
#         print(f"+++++++++++++ {user_name} Registered Successfully ++++++++++++++++")
#         messagebox.showinfo("Success", f"{user_name} has been registered successfully")
#     else:
#         messagebox.showwarning("Cancelled", "Registration cancelled.")


def infer(tensor):

    if source_type=="stream":
        audio_strength = torch.mean(torch.abs(tensor.type(torch.float)))
        # print("strengh: ",audio_strength)
        if(audio_strength < torch.tensor(500)):
            print("slience", )
            return 0
    elif source_type=="file":
        tensor = torchaudio.load(src_file)[0]


    # tensor = F_audio.resample(tensor, 16000, 8000)
    audio_data = preprocess(tensor)
    out_kw, out_id, map_kw, map_s = model(x=audio_data)
    print("+++++++++++++ 检测到说话 ++++++++++++++++")
    # map_s = F.avg_pool2d(map_s, (1, 13))
    # map_s = gap(map_s)
    # diff = torch.norm((map_s - anchors[0] ), p='fro') #/ ( map_s.shape[1]*map_s.shape[3])
    kw_prob = out_kw.max(dim=1).values.item()
    print(f"{WORD_LIST[out_kw.argmax(dim=1)]}: {kw_prob:.2f}" )
    # if kw_prob > 0.9:
    #     print(f"{WORD_LIST[out_kw.argmax(dim=1)]}: {kw_prob:.2f}" )
    # else:
    #     print("unknown keyword")
    sims=[]
    for name,anchor in anchors:
        # sim = torch.cosine_similarity(map_s.view(1,-1), anchor.view(1,-1))
        sim = F.cosine_similarity(map_s.view(map_s.size(0), -1), anchor)
        # print(sim)
        # if(sim > 0.995):
        print(f"{name}  similarity:{sim.item():.5f}", )
        # else:
        #     print("unknown speaker")

    # if out_id.max(dim=1).values >= 0.5:
    #     print(f"The Speaker is {SPEAKER_LIST[out_id.argmax(dim=1)]}")
    # else:
    #     print("Unknown Speaker")
    print("++++++++++++++++++++++++++++++++++++")
    # print(out_kw.argmax(dim=1),out_id.argmax(dim=1))
    # print(label_kw,label_id)
    # print(out_kw.max(dim=1).values, out_id.max(dim=1).values)





def record_and_register():
    print("+++++++++++++ Register ++++++++++++++++")
    user_name = simpledialog.askstring("Input", "Enter your name:", parent=root)
    if(not user_name):
        messagebox.showwarning("Cancelled", "Registration cancelled.")
        return
    SPEAKER_LIST.append(user_name)
    sub_anchors=[]
    for i in range(3):
        if source_type=="stream":
            print("say something in 2 s")
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            stream.stop_stream()
            stream.close()
            p.terminate()
            audio_data = b''.join(frames)
            tensor = torch.tensor(np.frombuffer(audio_data, dtype=np.int16), dtype=torch.float32)
        elif source_type=="file":
            tensor = torchaudio.load(anchors_file[i])[0]
        audio_data = preprocess(tensor)
        out_kw, out_id, map_kw, map_s = model(x=audio_data)
        # anchor = map_s
        # anchor = F.avg_pool2d(anchor, (1, 13))
        sub_anchors.append(map_s.view(map_s.size(0), -1))
    anchor = sum(sub_anchors)/len(sub_anchors)
    anchors.append([user_name,anchor])
    print(f"+++++++++++++ {user_name} Registered Successfully ++++++++++++++++")
    messagebox.showinfo("Success", f"{user_name} has been registered successfully")

def continuous_recording():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    while test_active:
        data = stream.read(CHUNK)
        audio_buffer.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

def continuous_inference():
    while test_active:
        if len(audio_buffer) == audio_buffer.maxlen:
            audio_data = b''.join(audio_buffer)
            tensor = torch.tensor(np.frombuffer(audio_data, dtype=np.int16), dtype=torch.float32)
            infer(tensor)  # 假设infer函数已定义
            sleep(2)
            # print("start")

# GUI操作
def start_register():

    record_thread = threading.Thread(target=record_and_register)
    record_thread.start()
    # thread.quit()
    # record_thread.wait()


def start_test():
    global test_active
    test_active = True
    record_thread = threading.Thread(target=continuous_recording)
    infer_thread = threading.Thread(target=continuous_inference)
    record_thread.start()
    infer_thread.start()

def stop_test():
    global test_active
    test_active = False




# if __name__ == "__main__":
    # 创建GUI界面
# GUI 按钮和布局
root = tk.Tk()
root.title("Test")
root.geometry("500x300")
style = {'padx': 5, 'pady': 5}
register_button = tk.Button(root, text="Register", command=lambda: record_and_register(), **style)
register_button.pack(pady=20)

test_button = tk.Button(root, text="Test", command=start_test, **style)
test_button.pack(pady=20)

stop_button = tk.Button(root, text="Stop Test", command=stop_test, **style)
stop_button.pack(pady=20)

root.mainloop()


    # reg_name = "data_register/B001_2.wav"
    # register(sample_name=reg_name)
    # infer()
