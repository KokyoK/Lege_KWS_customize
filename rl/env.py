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
import gym
from gym import spaces



# 环境定义
class EarlyExitEnv(gym.Env):
    def __init__(self, neural_network, dataloader, constraint):
        super(EarlyExitEnv, self).__init__()
        self.neural_network = neural_network  # 神经网络模型
        self.dataloader = dataloader  # 数据集
        self.iterator = iter(dataloader)  # 创建迭代器
        self.constraint = constraint  # 延迟约束
        self.action_space = spaces.Discrete(neural_network.num_exits)  # 动作空间：选择哪个 exit
        self.observation_space = spaces.Dict({
                                    "feature": spaces.Box(low=-np.inf, high=np.inf, shape=(np.array([1, 40, 1, 101]))),
                                    "avg_latency": spaces.Box(low=float(0), high=float(1),shape=(np.array([1]))),
                                    # "":
                                              })
        # })
        # (spaces.Box(low=-np.inf, high=np.inf, shape=(np.array([1, 40, 1, 101]))),
        #  spaces.Box(low=0, high=1, shape=1))
        self.state = None
        self.label = None
        self.total_hit = 0
        self.total_inference_time = 0  # 总推断时间
        self.sample_count = 0  # 推断的样本数量
    def calculate_accuracy(self,p):
        if (not torch.is_tensor(p)):
            p=p.value
        return float((p.argmax(dim=1) == self.label).item())
    def step(self, action):
        # 执行动作，这里的动作是选择 exit
        probabilities = self.neural_network.forward(x=self.state["feature"], exit_point=action)
        accuracy = self.calculate_accuracy(probabilities)  # 计算准确度
        energy = calculate_energy(self.neural_network.flops, action)  # 计算能量消耗
        inference_time = self.neural_network.flops[action]  # 获取推断时间

        self.total_hit += accuracy
        self.total_inference_time += inference_time
        self.sample_count += 1
        average_inference_time = self.total_inference_time / self.sample_count  # 计算平均推断时间

        # 计算奖励，这里简化为准确率与能量消耗的函数
        reward = accuracy / energy

        # 获取下一个样本
        try:
            next_feat, self.label = next(self.iterator)
            next_state ={"feature": next_feat,  # 或者其他适合你应用的初始化方式
                        "avg_latency": torch.Tensor([average_inference_time])} # 假设初始平均延迟为0
            done = False
            self.state = next_state
        except StopIteration:
            done = True  # 如果数据集结束，则完成

        return self.state, reward, done, {}

    def reset(self):
        # 重置环境
        self.iterator = iter(self.dataloader)
        feat, self.label = next(self.iterator)
        self.total_inference_time = 0
        self.state = {"feature": feat,  # 或者其他适合你应用的初始化方式
                    "avg_latency": torch.Tensor([0])} # 假设初始平均延迟为0
        self.sample_count = 0
        self.total_hit = 0
        return self.state

    def render(self, mode='human'):
        # 可视化（如果需要）
        pass

    def close(self):
        pass


# 定义能量消耗计算函数
def calculate_energy(flops, exit_point):
    # 假设每个FLOP消耗的能量是一个常数
    energy_per_flop = 0.9# 可以根据实际情况调整
    return flops[exit_point] * energy_per_flop


