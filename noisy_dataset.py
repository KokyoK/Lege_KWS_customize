import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# from numpy.lib.function_base import i0
import torch.utils.data as data
import torch.nn.functional as F
import torchaudio.functional as F_audio
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torchaudio
import random
import csv
import torch
torch.manual_seed(42)
random.seed(42)
import math

train_noise_count = 8
valid_noise_count = 8
test_noise_count = 8
dataset_folder = "dataset"

# if torch.cuda.is_available():
#     torch.cuda.set_device(geforce_rtx_3060_xc)

def fetch_speaker_list(ROOT_DIR, WORD_LIST):
    speaker_list = []
    if "huawei" in ROOT_DIR:
        available_words = os.listdir(ROOT_DIR)  # 列出原数据集的words
        for i, word in enumerate(available_words):
            if os.path.isdir(os.path.join(ROOT_DIR, available_words[i])):  # 排除.DS_store这种文件
                if (word in WORD_LIST):
                    for wav_file in os.listdir(ROOT_DIR + word):
                        if wav_file.endswith(".wav"):
                            id = wav_file.split("_", 1)[0]
                            if (id not in speaker_list):
                                speaker_list.append(id)
    elif "lege" in ROOT_DIR:
        ROOT_DIR = "dataset_lege/lege_origin/" 
        available_words = os.listdir(ROOT_DIR)  # 列出原数据集的words
        for i, word in enumerate(available_words):
            if os.path.isdir(os.path.join(ROOT_DIR,available_words[i])):    # 排除.DS_store这种文件
                if (word in WORD_LIST):
                    for wav_file in os.listdir(ROOT_DIR + word):
                        if wav_file.endswith(".wav"):
                            id = wav_file.split("_", 1)[0]
                            if (id not in speaker_list):
                                speaker_list.append(id)
            # else:

    elif ROOT_DIR == f"{dataset_folder}/google_origin/" or ROOT_DIR == f"{dataset_folder}/google_noisy/NGSCD/" or ROOT_DIR == f"{dataset_folder}/google_noisy/NGSCD_SPEC/" :
        ROOT_DIR = f"{dataset_folder}/google_origin/" 
        available_words = os.listdir(ROOT_DIR)  # 列出原数据集的words
        for i, word in enumerate(available_words):
            if (word in WORD_LIST):
                for wav_file in os.listdir(ROOT_DIR + word):
                    if wav_file.endswith(".wav"):
                        id = wav_file.split("_", 1)[0]
                        if (id not in speaker_list):
                            speaker_list.append(id)
    return speaker_list
import pandas as pd
import os
import random

import pandas as pd
import os
import random

def merge_noisy_datasets(csv_lists_path):
    # Predefined SNR lists
    snrs_tr = ['0', '5', '10', '15', '20','-']  # Training and validation SNRs
    snrs_te = ['-10', '-5', '0', '5', '10', '15', '20', '-']  # Test SNRs

    # Function to merge and add columns to CSV files
    def merge_and_add_columns(file_type, snrs):
        frames = []
        for i in range(1, 9):
            file_path = os.path.join(csv_lists_path, f'{file_type}_clean{i}.csv')
            df = pd.read_csv(file_path)
            # Assign random noise type and noise level to each row
            df['noise'] = df.apply(lambda x: i, axis=1)
            df['noise_level'] = df.apply(lambda x: random.choice(snrs), axis=1)
            frames.append(df)
        merged_df = pd.concat(frames)
        return merged_df

    # Merge and add columns to the CSV files
    train_df = merge_and_add_columns('train', snrs_tr)
    valid_df = merge_and_add_columns('valid', snrs_tr)
    test_df = merge_and_add_columns('test', snrs_te)

    # Save the new datasets
    train_df.to_csv(os.path.join(f'{dataset_folder}/lege_noisy/split', 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(f'{dataset_folder}/lege_noisy/split', 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(f'{dataset_folder}/lege_noisy/split', 'test.csv'), index=False)

    return train_df, valid_df, test_df





class CsvLogger:
    def __init__(self, filename, head):
        self.filename = filename
        # 初始化时创建文件并写入标题
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(head)  # 举例的标题行

    def log(self, data):
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header line
        chunk_list = []
        # A temporary list to store up to three rows
        temp_chunk = []
        for row in reader:
            temp_chunk.append(tuple(row))
            if len(temp_chunk) == 3:
                chunk_list.append(tuple(temp_chunk))
                temp_chunk = []  # Reset the temporary list after adding to chunk_list
        if temp_chunk:  # Add any remaining rows that didn't make a full chunk of three
            chunk_list.append(tuple(temp_chunk))

    return chunk_list

def create_csv(root_dir, word_list,speaker_list):
    train_csv = CsvLogger(filename=f'{dataset_folder}/split/train.csv', head=["path","kw","id"])
    valid_csv = CsvLogger(filename=f'{dataset_folder}/split/valid.csv', head=["path","kw","id"])
    test_csv = CsvLogger(filename=f'{dataset_folder}/split/test.csv', head=["path","kw","id"])
    
    
    
    train, dev, test = split_dataset(root_dir, word_list, speaker_list)
    ap = AudioPreprocessor()
    train_data = SpeechDataset(train, "train", ap, word_list, speaker_list)
    dev_data = SpeechDataset(dev, "train", ap, word_list, speaker_list)
    test_data = SpeechDataset(test, "train", ap, word_list, speaker_list)

    train_trips = generate_triplets(train_data)
    # train_trip_dataset = TripletSpeechDataset(train_trip, "train", ap, word_list, speaker_list)
    valid_trips = generate_triplets(dev_data)
    # dev_trip_dataset = TripletSpeechDataset(dev_trip, "train", ap, word_list, speaker_list)
    test_trips = generate_triplets(test_data)
    # test_trip_dataset = TripletSpeechDataset(test_trip, "train", ap, word_list, speaker_list)
    for train_trip in train_trips:
        for sample in train_trip:
            train_csv.log(sample)
    
    for valid_trip in valid_trips:
        for sample in valid_trip:
            valid_csv.log(sample)
    
    for test_trip in test_trips:
        for sample in test_trip:
            test_csv.log(sample)   
            
def get_all_data_length(root_dir):          # for debug
    sample_count = 0
    for available_words in os.listdir(root_dir):  #
        if os.path.isdir(root_dir+available_words):
            sample_count += len(os.listdir(root_dir+available_words))
    print(sample_count)
    return sample_count



# 在train_set中的speaker 不会出现在valid_set和test_set里
def split_dataset(root_dir, word_list, speaker_list, split_pct=[0.8, 0.1, 0.1]):
    if sum(split_pct) != 1:
        raise ValueError("Split percentages must sum to 1")
    unknown_list = []
    train_set = []
    dev_set = []
    test_set = []

    # Shuffle and split the speaker list into two groups
    # random.shuffle(speaker_list)
    # speaker_list = speaker_list.sort()
    speaker_list = sorted(speaker_list, reverse=False)
    n_train_speakers = int(len(speaker_list) * split_pct[0])
    train_speakers = speaker_list[:n_train_speakers]
    other_speakers = speaker_list[n_train_speakers:]
    

    # Process each word
    for word in os.listdir(root_dir):
        if word in word_list and os.path.isdir(os.path.join(root_dir, word)):
            # Process each speaker
            for speaker in speaker_list:
                speaker_samples = []

                for wav_file in os.listdir(os.path.join(root_dir, word)):
                    if wav_file.endswith(".wav") and wav_file.split("_", 1)[0] == speaker:
                        speaker_samples.append((os.path.join(root_dir, word, wav_file), word, speaker))

                # Split speaker_samples into train/dev/test sets
                if speaker in train_speakers:
                    train_set += speaker_samples
                else:
                    # Split between dev and test sets
                    n_dev = int(len(speaker_samples) * split_pct[1] / (split_pct[1] + split_pct[2]))
                    dev_set += speaker_samples[:n_dev]
                    test_set += speaker_samples[n_dev:]


    if "unknown" in word_list:
        random.shuffle(unknown_list)
        n_samples = len(unknown_list)
        n_train = int(n_samples * split_pct[0])
        n_dev = int(n_samples * split_pct[1])
        train_set += unknown_list[:n_train]
        dev_set += unknown_list[n_train:n_train + n_dev]
        test_set += unknown_list[n_train + n_dev:]

    if "silence" in word_list:
        silence_samples = [os.path.join(root_dir, "_background_noise_", wav_file) for wav_file in os.listdir(os.path.join(root_dir, "_background_noise_")) if wav_file.endswith(".wav")]
        n_samples = len(silence_samples)
        n_train = int(n_samples * split_pct[0])
        n_dev = int(n_samples * split_pct[1])
        train_set += silence_samples[:n_train]
        dev_set += silence_samples[n_train:n_train + n_dev]
        test_set += silence_samples[n_train + n_dev:]

    random.shuffle(train_set)
    random.shuffle(dev_set)
    random.shuffle(test_set)
    # print(train_set,dev_set)
    return train_set, dev_set, test_set


class SpeechDataset(data.Dataset):

    def __init__(self, data_list, dataset_type, transforms, word_list, speaker_list, is_noisy=False, is_shift=False,
                 sample_length=16000):
        """ types include [TRAIN, DEV, TEST] """
        self.data_list = data_list
        self.dataset_type = dataset_type
        self.is_noisy = is_noisy
        self.is_shift = is_shift
        self.sample_length = sample_length
        self.transforms = transforms
        self.word_list = word_list
        self.speaker_list = speaker_list

    def shift_audio(self, audio_data, max_shift=160):
        """ Shifts audio.
        """
        shift_val = random.randint(-max_shift, max_shift)
        zero_fill = torch.zeros(audio_data.shape[0], abs(shift_val))
        if (shift_val < 0):
            audio_data = torch.cat((audio_data[:, 0:(audio_data.shape[1] + shift_val)], zero_fill), 1)
        else:
            audio_data = torch.cat((zero_fill, audio_data[:, shift_val:]), 1)
        return audio_data

    def load_data(self, data_element):
        """ Loads audio, shifts data and adds noise. """
        # print(data_element)
        wav_data = torchaudio.load(data_element[0])[0]
        wav_data = F_audio.resample(wav_data, 16000, 8000)  # @NOTE: 下采样到8000，部署的时候改原采样率

        # Background noise used for silence needs to be shortened to 1 second.
        if (data_element[1] == "silence"):
            slice_idx = random.randint(0, wav_data.view(-1).shape[0] - self.sample_length - 1)
            amplitude = random.random()
            out_data = amplitude * wav_data[:, slice_idx:slice_idx + self.sample_length]
        else:
            out_data = 1 * wav_data
        # print(out_data.shape)
        
       
        data_len = 16000
        # Pad smaller audio files with zeros to reach 1 second (16_000 samples)
        if (out_data.shape[1] < data_len):
            out_data = F.pad(out_data, pad=(0, (data_len - out_data.shape[1])), mode='constant', value=0)

        # Clip larger audio files with zeros to reach 1 second (16_000 samples)
        if (out_data.shape[1] > data_len):
            t = out_data.shape[1] - data_len
            out_data = out_data[:,t:data_len+t]
            
        # print(out_data.shape)
        # Adds audio shift (upto 100 ms)
        if self.is_shift:
            out_data = self.shift_audio(out_data)

        # Add random noise
        if self.is_noisy:
            out_data += 0.01 * torch.randn(out_data.shape)
        
        return (out_data, data_element[1],data_element[2])




    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # print(self.data_list[idx])
        cur_element = self.load_data(self.data_list[idx])
        cur_element = (cur_element[0], self.word_list.index(cur_element[1]), self.speaker_list.index(cur_element[2]))
        return self.transforms(cur_element)


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


import itertools

def generate_triplets(dataset, num_anchors_per_sample=5):
    """
    Generates a list of triplets for training with triplet loss.
    Each triplet consists of an anchor, a positive, and a negative sample.
    Anchor and positive are from the same speaker but different words,
    while anchor and negative are from different speakers.

    Args:
        dataset (SpeechDataset): The dataset to generate triplets from.
        num_anchors_per_sample (int): Number of times each sample is used as an anchor.

    Returns:
        list of tuples: List containing triplets in the form (anchor, positive, negative).
    """
    triplets = []

    for anchor in dataset.data_list:
        anchor_speaker = anchor[2]  # Extracting speaker ID of the anchor

        # Selecting samples with the same speaker but different word for positive
        positive_candidates = [item for item in dataset.data_list if item[2] == anchor_speaker and item[1] != anchor[1]]
        # Selecting samples with different speakers for negative
        negative_candidates = [item for item in dataset.data_list if item[2] != anchor_speaker]

        if len(positive_candidates) == 0 or len(negative_candidates) == 0:
            continue

        # Randomly choosing different positives and negatives for each anchor
        for _ in range(num_anchors_per_sample):
            positive = random.choice(positive_candidates)
            negative = random.choice(negative_candidates)
            triplets.append((anchor, positive, negative))

    return triplets
# Example of how to use the function
class TripletSpeechDataset(data.Dataset):
    def __init__(self, triplet_list, dataset_type, transforms, word_list, speaker_list,
                 sample_length=16000):
        self.triplet_list = triplet_list
        self.transforms = transforms
        self.dataset_type = dataset_type
        self.sample_length = sample_length
        self.transforms = transforms
        self.word_list = word_list
        self.speaker_list = speaker_list   
        for j in range(len(self.triplet_list)):
            # triplet = triplet_list[j]
            anchor, positive, negative = self.triplet_list[j]
            triplet =  [anchor,positive,negative]
            for i in range(len(triplet)):
                spec, clean_spec, cur_element = self.load_data(triplet[i])
                triplet[i] = (spec,clean_spec, [self.word_list.index(cur_element[0]), self.speaker_list.index(cur_element[1]),cur_element[2],cur_element[3]])
            self.triplet_list[j] = triplet   
 
    def __len__(self):
        return len(self.triplet_list)

    def __getitem__(self, idx):

        triplet = self.triplet_list[idx]
        return triplet



        # return anchor_data, positive_data, negative_data

    def load_data(self, data_element):
        """ Loads audio, shifts data and adds noise. """
        # print(data_element) # ('up/888a0c49_nohash_3.wav', 'up', '888a0c49', '1', '20')
        if dataset_folder == "dataset_lege":
            dataset_name = "lege"
        elif dataset_folder == "dataset":
            dataset_name = "google"
        datapath_root = f"{dataset_folder}/{dataset_name}_noisy/NGSCD_SPEC/"
        # data_path = self.dataset_type+"/"+ 
        if data_element[4]=='-':
            data_path = f"clean{data_element[3]}"
        else:
            data_path = f"N{data_element[3]}_SNR{data_element[4]}"
        data_path = datapath_root + self.dataset_type+"/"+ data_path + "/" +data_element[0].replace("/","_")

        cleanpath_root = f"{dataset_folder}/{dataset_name}_origin_SPEC/"
        clean_path =  cleanpath_root+data_element[0]
        
        # 读取 wav
        # out_data = self.process_audio(data_path=data_path)
        # clean_data = self.process_audio(data_path=clean_path)
        
        # 读取 spec
        
        out_data = torch.load(data_path.replace(".wav",".pt"))
        clean_data = torch.load(clean_path.replace(".wav",".pt"))
        # print(data_element) # ('up/888a0c49_nohash_3.wav', 'up', '888a0c49', '1', '-')
        return (out_data,clean_data, data_element[1:])
    
    def process_audio(self, data_path):
        wav_data = torchaudio.load(data_path)[0]
        wav_data = F_audio.resample(wav_data, 16000, 8000)  # 认为是一秒的数据

        out_data = 1 * wav_data
        # print(out_data.shape)

        data_len = 16000
        # Pad smaller audio files with zeros to reach 1 second (16_000 samples)
        if (out_data.shape[1] < data_len):
            out_data = F.pad(out_data, pad=(0, (data_len - out_data.shape[1])), mode='constant', value=0)

        # Clip larger audio files with zeros to reach 1 second (16_000 samples)
        if (out_data.shape[1] > data_len):
            t = out_data.shape[1] - data_len
            out_data = out_data[:, t:data_len + t]

        # to spectrum
        out_data = self.transforms(out_data)
        return out_data
        
def get_loaders( root_dir, word_list,speaker_list,):
    train, dev, test = split_dataset(root_dir, word_list, speaker_list)
    ap = AudioPreprocessor()
    
    split_root = f'{root_dir.replace("NGSCD_SPEC/","")}split/'
    
    train_trips = []
    valid_trips = []
    test_trips = []
    # for i in range(train_noise_count):
    #     train_trips.append(read_csv(split_root+"train.csv"))    
        
    train_trip = read_csv(split_root+"train.csv")
    valid_trip = read_csv(split_root+"valid.csv")
    test_trip = read_csv(split_root+"test.csv")

    bs = 32

    train_trip_dataset = TripletSpeechDataset(train_trip, "Train", ap, word_list, speaker_list)
    # dev_trip = generate_triplets(dev_data)
    valid_trip_dataset = TripletSpeechDataset(valid_trip, "Valid", ap, word_list, speaker_list)
    # test_trip = generate_triplets(test_data)
    test_trip_dataset = TripletSpeechDataset(test_trip, "Test", ap, word_list, speaker_list)

    train_trip_loader = data.DataLoader(train_trip_dataset, batch_size=bs, shuffle=True)
    valid_trip_loader = data.DataLoader(valid_trip_dataset, batch_size=bs, shuffle=True)
    test_trip_loader = data.DataLoader(test_trip_dataset, batch_size=bs, shuffle=True)

    return [train_trip_loader, valid_trip_loader, test_trip_loader]

   
if __name__ == "__main__":
    # Test example
    dataset_folder = "dataset_lege"
    root_dir = f"{dataset_folder}/lege_noisy/NGSCD/"
    word_list = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
    speaker_list = fetch_speaker_list(root_dir,word_list)
    print("num speakers: ", len(speaker_list))
    loaders = get_loaders(root_dir, word_list, speaker_list)
    torch.save(loaders,"loaders/loaders_lege.pth")
    # torch.save(self.state_dict(), "saved_model/" + name)
    # @todo: data preparation

    # root_dir =  f"{dataset_folder}/google_noisy/NGSCD/"
    # word_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    # speaker_list = fetch_speaker_list(root_dir,word_list)
    # print("num speakers: ", len(speaker_list))
    # loaders = get_loaders(root_dir, word_list, speaker_list)
    
    

    
    # # NOTE 生成数据集 merge后的list
    # csv_lists_path = 'dataset_lege/NoisyLEGE/csvLists/'
    # train_df, valid_df, test_df = merge_noisy_datasets(csv_lists_path)
    # print(train_df.head())
    # print(valid_df.head())
    # print(test_df.head())
   