
import pyaudio
import wave
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torchaudio.functional as F_audio
import torchaudio
import torch
import model as md
from nn_main import WORD_LIST,SPEAKER_LIST
import torch.nn.functional as F
import random
import time
torch.manual_seed(42)


model = md.TCResNet8(k=1, n_mels=40, n_classes=len(WORD_LIST),n_speaker=len(SPEAKER_LIST))
model.load()
model.eval()
model.mode = "eval"
meta = {"prev_frame":torch.tensor([0])}



def shift_audio( audio_data, max_shift=160):
    """ Shifts audio.
    """
    shift_val = random.randint(-max_shift, max_shift)
    zero_fill = torch.zeros(audio_data.shape[0], abs(shift_val))
    if (shift_val < 0):
        audio_data = torch.cat((audio_data[:, 0:(audio_data.shape[1] + shift_val)], zero_fill), 1)
    else:
        audio_data = torch.cat((zero_fill, audio_data[:, shift_val:]), 1)
    return audio_data



def callback(in_data, frame_count, time_info, status):
    # print(time_info,status)
    half_data = torch.frombuffer(in_data, dtype=torch.int16)
    audio_data = meta["prev_frame"] + half_data
    meta["prev_frame"] = half_data
    audio_strength = torch.mean(torch.abs(audio_data.type(torch.float)))
    print("strengh: ",audio_strength)
    if(audio_strength < torch.tensor(1000)):
        print("slience")
        return (in_data, pyaudio.paContinue)
    else:
        audio_data = audio_data / (2 ** 15)
        audio_data = torch.reshape(audio_data, [1, -1])
        audio_feat = extract_feature(audio_data)
        # print("feat",audio_feat)
        infer(audio_feat, model)
        return (in_data, pyaudio.paContinue)

def audio():
    p = pyaudio.PyAudio()

    # FORMAT = p.get_format_from_width(wf.getsampwidth())
    # CHANNELS = wf.getnchannels()
    # RATE = wf.getframerate()

    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # 声道数
    RATE = 8000  # 采样率
    CHUNK = 8000


    # WAVE_OUTPUT_FILENAME = "output.wav"

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    stream_callback=callback,
                    frames_per_buffer=CHUNK)

    print("* recording")
    stream.start_stream()
    while stream.is_active():
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()

    p.terminate()

    frames = []
    prev_data = []
    # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #     data = stream.read(STEP)
    #     print("data", len(data))
        # data = wf.readframes(CHUNK)
        # if (i == 0):
        #     step_data = stream.read(STEP)
        #     prev_data = step_data
        #     continue
        # else:
        #     step_data = stream.read(STEP)
        #     print("data", len(step_data))
        #     data = prev_data[CHUNK-STEP:CHUNK] + step_data
        #     prev_data = data
        #     # print(len(data))

        # audio_data = np.frombuffer(data, dtype=int8)    # audio data length = 1024


    print("* done recording")


    # a = torchaudio.load(sample)
    # print(a[0])



def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=False)

def extract_feature(waveform):
    is_shift = False
    is_noisy = False
    # is_shift = True
    # is_noisy = True
    # wav_data = F_audio.resample(waveform, 48000, 8000)  # @NOTE: 下采样到8000，部署的时候改原采样率
    # wav_data = 1 * wav_data
    wav_data = waveform
    data_len = 16000

    # Pad smaller audio files with zeros to reach 1 second (16_000 samples)
    if (wav_data.shape[1] < data_len):
        wav_data = F.pad(wav_data, pad=(0, (data_len - wav_data.shape[1])), mode='constant', value=0)

    # Clip larger audio files with zeros to reach 1 second (16_000 samples)
    if (wav_data.shape[1] > data_len):
        wav_data = wav_data[:, :data_len]

    if is_shift:
        wav_data = shift_audio(wav_data)

    # Add random noise
    if is_noisy:
        wav_data += 0.01 * torch.randn(wav_data.shape)

    #################################### extract feat ##########################
    trans_spectrogram = nn.Sequential(
       # torchaudio.transforms.Resample(48000, 16000),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=480,  # window length = 30,
            hop_length=160,  # stride = 10
            f_min=0,
            f_max=8000,
            n_mels=40,
            window_fn=torch.hann_window
        ),
        torchaudio.transforms.AmplitudeToDB()
    )
    out_data = trans_spectrogram(wav_data)
    out_data = out_data.view(1,out_data.shape[1], out_data.shape[0], out_data.shape[2])

    return out_data


def infer(audio_data,model):
    word_list = WORD_LIST
    path, output = model(audio_data)
    output_label = word_list[torch.argmax(output, 1)]
    print(torch.max(output) )
    if(torch.max(output) >= 0.95 ):
        print("inferrd label: ", output_label)
    else:
        print("not recognized")






if __name__ == '__main__':
    # Monitor()
    audio()



"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).

This is the callback (non-blocking) version.
"""








