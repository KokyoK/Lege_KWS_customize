import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from thop import profile
from collections import deque
import matplotlib.pyplot as plt
from model_quantize import *
from model import *
import model_quantize as mdq
import speech_dataset as sd
from env import *
import gym
from gym import spaces
import brevitas.nn as qnn
from brevitas.quant import Int32Bias

class DQN(nn.Module):
    def __init__(self, feature_shape, action_size):
        super(DQN, self).__init__()
        # self.quant_inp = qnn.QuantIdentity(bit_width=qa, return_quant_tensor=True)
        # self.conv0 = qnn.QuantConv2d(in_channels=40, out_channels=16, kernel_size=(1, 3), stride=1,
        #                                     bias=True,  weight_bit_width=qw, padding=(0, 1), bias_quant=Int32Bias)
        # self.relu0 = qnn.QuantReLU(bit_width=qw, return_quant_tensor=True)
        # self.conv1 = qnn.QuantConv2d(in_channels=16, out_channels=32, kernel_size=(1, 5), stride=3,
        #                              bias=True, weight_bit_width=qw,  bias_quant=Int32Bias)
        # self.relu1 = qnn.QuantReLU(bit_width=qw, return_quant_tensor=True)
        # self.fc0 = qnn.QuantLinear(33, 32, bias=False, weight_bit_width=qw)
        # self.relu3 = qnn.QuantReLU(bit_width=qw, return_quant_tensor=True)
        # self.fc1 = qnn.QuantLinear(32, action_size, bias=False, weight_bit_width=qw)
        # self.relu4 = qnn.QuantReLU(bit_width=qw, return_quant_tensor=True)
        self.conv0 = nn.Conv2d(in_channels=40, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1),bias=True)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 5), stride=3,bias=True)
        self.relu1 = nn.ReLU()
        self.fc0 = nn.Linear(32 + 1, 32)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(32, action_size)
        self.relu4 = nn.ReLU()


    def forward(self, x):
        # feature = self.quant_inp(x['feature'])
        # avg_latency = self.quant_inp(x['avg_latency'])
        feature = x['feature']
        avg_latency = x['avg_latency']
        if avg_latency.shape==torch.Size([1]):
            avg_latency = avg_latency.unsqueeze(1)
        feature = self.relu0(self.conv0(feature))
        feature = self.relu1(self.conv1(feature))
        feature = F.max_pool2d(feature, kernel_size=(1, 33), stride=1)
        feature = feature.view(feature.shape[0], -1)  # 拉平

        # Concatenate the flattened feature tensor with avg_latency
        cat = torch.cat([feature, avg_latency], dim=1)

        cat = self.relu3(self.fc0(cat))
        action_values = self.relu4(self.fc1(cat))
        return action_values


# 代理（Agent）定义
class Agent:
    def __init__(self, feat_shape, action_size, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 learning_rate=0.001, batch_size=50, memory_size=100):
        self.feat_shape = feat_shape
        self.action_size = action_size
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon_start  # 探索率
        self.epsilon_end = epsilon_end  # 最小探索率
        self.epsilon_decay = epsilon_decay  # 探索率衰减
        self.batch_size = batch_size  # 批量大小
        self.memory = deque(maxlen=memory_size)  # 经验回放缓冲区
        self.model = DQN(feat_shape, action_size)  # DQN模型
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)  # 优化器

    # 存储经验
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 选择动作
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)  # 探索
        else:
            # 为字典中的每个张量添加一个批次维度
            state_dict = {k: v for k, v in state.items()}
            q_values = self.model(state_dict)
            return np.argmax(q_values.detach().numpy())

    # 经验回放
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # 分别处理字典中的每个键
        feature_states = np.vstack([s['feature'] for s in states])
        avg_latency_states = np.vstack([s['avg_latency'] for s in states])
        feature_next_states = np.vstack([s['feature'] for s in next_states])
        avg_latency_next_states = np.vstack([s['avg_latency'] for s in next_states])

        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        non_final_mask = torch.tensor(tuple(map(lambda s: s != 1, dones)), dtype=torch.bool)
        non_final_next_states = [next_states[i] for i in range(len(next_states)) if non_final_mask[i]]

        # 准备模型的输入
        state_input = {'feature': torch.FloatTensor(feature_states),
                       'avg_latency': torch.Tensor(avg_latency_states)}
        next_state_input = {'feature': torch.FloatTensor(feature_next_states),
                            'avg_latency': torch.Tensor(avg_latency_next_states)}

        Q_targets_next = torch.zeros(self.batch_size)
        Q_targets_next_non_final = self.model(next_state_input).detach().max(1)[0]
        Q_targets_next = torch.where(non_final_mask, Q_targets_next_non_final, Q_targets_next)

        # 计算Q目标值
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.model(state_input).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay  # 更新探索率

