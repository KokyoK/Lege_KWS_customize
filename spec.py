import os
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.functional as F_audio

class AudioPreprocessor():
    def __init__(self):
        self.spectrogram = nn.Sequential(
            # torchaudio.transforms.Resample(48000, 16000),
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

    def __call__(self, data):
        # print(data[0].shape)
        o_data = self.spectrogram(data)
        # print(o_data.shape)
        # o_data = self.mfcc(data[0])
        # print(o_data[0].shape)
        # Set Filters as channels
        o_data = o_data.view(o_data.shape[1], o_data.shape[0], o_data.shape[2])
        # print(o_data.shape,data[1])
        return o_data
    
def process_audio_files(root_dir, dest_dir):
    # 遍历根目录
    ap = AudioPreprocessor()
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):  # 假设语音文件是WAV格式
                audio_path = os.path.join(subdir, file)
                wav_data = torchaudio.load(audio_path)[0]
                
                # wav_data = F_audio.resample(wav_data, 16000, 8000)  # 认为是一秒的数据
                data_len = 16000
                out_data = wav_data
                # Pad smaller audio files with zeros to reach 1 second (16_000 samples)
                if (out_data.shape[1] < data_len):
                    out_data = F.pad(out_data, pad=(0, (data_len - out_data.shape[1])), mode='constant', value=0)
                # Clip larger audio files with zeros to reach 1 second (16_000 samples)
                if (out_data.shape[1] > data_len):
                    t = out_data.shape[1] - data_len
                    out_data = out_data[:, t:data_len + t]
   
                # to spectrum
                out_data = ap(out_data)
                
                              
                rel_path = os.path.relpath(audio_path, root_dir)  # 相对路径

                import os

    def __init__(self):
        self.spectrogram = nn.Sequential(
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

    def __call__(self, data):
        # print(data[0].shape)
        o_data = self.spectrogram(data)
        # print(o_data.shape)
        # o_data = self.mfcc(data[0])
        # print(o_data[0].shape)
        # Set Filters as channels
        o_data = o_data.view(o_data.shape[1], o_data.shape[0], o_data.shape[2])
        # print(o_data.shape,data[1])
        return o_data


# 保存带噪声数据的时频图
def process_audio_files(root_dir, dest_dir):
    # 遍历根目录
    ap = AudioPreprocessor()
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):  # 假设语音文件是WAV格式
                audio_path = os.path.join(subdir, file)
                rel_path = os.path.relpath(audio_path, root_dir)  # 相对路径
                wav_data = torchaudio.load(audio_path)[0]
                # wav_data = F_audio.resample(wav_data, 16000, 8000)  # 假设将采样率从16000降低到8000
                data_len = 16000  
                out_data = wav_data
                # Pad smaller audio files with zeros to reach 1 second (8_000 samples)
                if (out_data.shape[1] < data_len):
                    out_data = F.pad(out_data, pad=(0, (data_len - out_data.shape[1])), mode='constant', value=0)
                # Clip larger audio files with zeros to reach 1 second (8_000 samples)
                if (out_data.shape[1] > data_len):
                    t = out_data.shape[1] - data_len
                    out_data = out_data[:, t:data_len + t]
                # to spectrum
                out_data = ap(out_data)
                # 构造目标路径
                spec_path = os.path.join(dest_dir, rel_path)
                spec_dir = os.path.dirname(spec_path)
                os.makedirs(spec_dir, exist_ok=True)
                # 保存时频图数据
                spec_filename = spec_path.replace('.wav', '.pt')  # 将文件扩展名改为.pt
                torch.save(out_data, spec_filename)
        print(subdir)

# 保存原始数据的时频图            
def process_audio_files_exclude_prefix(root_dir, dest_dir, prefix='_'):
    # 遍历根目录
    ap = AudioPreprocessor()
    for subdir, dirs, files in os.walk(root_dir):
        if os.path.basename(subdir).startswith(prefix):
            continue  # 跳过以指定前缀开头的文件夹
        for file in files:
            if file.endswith('.wav'):  # 假设语音文件是WAV格式
                audio_path = os.path.join(subdir, file)
                rel_path = os.path.relpath(audio_path, root_dir)  # 相对路径
                wav_data = torchaudio.load(audio_path)[0]
                wav_data = F_audio.resample(wav_data, 16000, 8000)  # 假设将采样率从16000降低到8000
                data_len = 16000  
                out_data = wav_data
                # Pad smaller audio files with zeros to reach 1 second (8_000 samples)
                if (out_data.shape[1] < data_len):
                    out_data = F.pad(out_data, pad=(0, (data_len - out_data.shape[1])), mode='constant', value=0)
                # Clip larger audio files with zeros to reach 1 second (8_000 samples)
                if (out_data.shape[1] > data_len):
                    t = out_data.shape[1] - data_len
                    out_data = out_data[:, t:data_len + t]
                # to spectrum
                out_data = ap(out_data)
                # 构造目标路径
                spec_path = os.path.join(dest_dir, rel_path)
                spec_dir = os.path.dirname(spec_path)
                os.makedirs(spec_dir, exist_ok=True)
                # 保存时频图数据
                spec_filename = spec_path.replace('.wav', '.pt')  # 将文件扩展名改为.pt
                torch.save(out_data, spec_filename)
        print(subdir)


# if __name__ == '__main__':
#     # root_dir = 'dataset/google_origin'
#     # dest_dir = 'dataset/google_origin_SPEC'

#     root_dir = 'dataset_lege/lege_origin'
#     dest_dir = 'dataset_lege/lege_origin_SPEC'

#     process_audio_files_exclude_prefix(root_dir, dest_dir)
    

    
if __name__ == '__main__':
    root_dir = 'dataset/google_noisy/NGSCD'
    dest_dir = 'dataset/google_noisy/NGSCD_SPEC'

    # root_dir = 'dataset_lege/lege_noisy/NGSCD'
    # dest_dir = 'dataset_lege/lege_noisy/NGSCD_SPEC'

    process_audio_files(root_dir, dest_dir)


               