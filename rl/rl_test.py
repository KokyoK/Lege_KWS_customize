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
from agent import *
from env import *
import gym
from gym import spaces
import brevitas.nn as qnn
from brevitas.quant import Int32Bias




# def cal_flops():
#     input = torch.randn(1, 40, 1, 101)
#     # 使用thop计算FLOPs和参数数量
#     flops, params = profile(ee_model, inputs=(input,))
#     print(f"FLOPs: {flops}") #  634800.0  1503888
#     print(f"Params: {params}")
#
#     input2 = {"feature": input,  # 或者其他适合你应用的初始化方式
#                 "avg_latency": torch.Tensor([0.5])} #
#     flops, params = profile(DQN(feat_shape, action_size), inputs=(input2,))
#     print(f"FLOPs: {flops}")        # 279520
#     print(f"Params: {params}")      # 5862


def run():
    root_dir = "../dataset/lege/"
    word_list = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
    speaker_list = []
    loaders = sd.kws_loaders(root_dir, word_list, speaker_list)
    # ee_model = QuantizedTCResNet8(1, 40, 10)
    ee_model = TCResNet8(1, 40, 10)
    ee_model.load("../saved_model/lege_ee_float_96.333.pt")
    ee_model.eval()

    env = EarlyExitEnv(ee_model, loaders[1], constraint=0.8)
    action_size = env.action_space.n  # 2, 从哪个exit出
    feat_shape = env.observation_space.spaces['feature'].shape  # 获取状态的实际形状
    # input_size = state_shape[1] * state_shape[2] * state_shape[3]  # 降维后的输入大小
    agent = Agent(feat_shape, action_size)  # 调整 Agent 的初始化

    episodes = 200  # 训练回合数
    scores = []  # 用于记录每个回合的总奖励
    average_inference_times = []  # 用于记录每个回合的平均推断时间

    ########## cal flops ##########
    # cal_flops()
    #################################
    for e in range(episodes):
        state = env.reset()
        # state = np.reshape(state, [1, input_size])  # 降维处理
        score = 0  # 初始化当前回合的总奖励
        total_inference_time = 0  # 初始化当前回合的总推断时间

        for time in range(300):  # 每回合最大步数
            # if (not env.observation_space.spaces['avg_latency'].shape[0] == 1):
            # if time == 65:
            #     print(time)
            action = agent.act(state)  # agent 做动作
            next_state, reward, done, _ = env.step(action)
            inference_time = env.neural_network.flops[action]  # 获取推断时间
            total_inference_time += inference_time  # 累加推断时间
            # reward = reward if not done else -10  # 调整奖励
            score += reward  # 累加奖励
            # next_state = np.reshape(next_state, [1, input_size])  # 降维处理
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # print(time)
            if done:
                average_inference_time = total_inference_time / time  # 计算平均推断时间
                average_inference_times.append(average_inference_time)  # 记录平均推断时间
                if average_inference_time > env.constraint:
                    score -= 1000 * (average_inference_time - env.constraint)  # 如果平均推断时间超过约束，则给予负奖励
                scores.append(score)  # 记录当前回合的总奖励
                print(
                    f"Episode: {e}/{episodes}, Score: {score:.3f}, Avg ACC: {env.total_hit / time * 100:.2f},Avg Time: {average_inference_time:.4f}, Epsilon: {agent.epsilon:.4}")
                break

            if len(agent.memory) > agent.batch_size:
                agent.replay()  # 经验回放

    # # 绘制学习过程
    # plt.plot(scores)
    # plt.title('DQN Learning Process')
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.show()
    #
    # # 绘制平均推断时间
    # plt.plot(average_inference_times)
    # plt.title('Average Inference Time per Episode')
    # plt.xlabel('Episode')
    # plt.ylabel('Average Inference Time (s)')
    # plt.show()

    # 测试模型
    # 测试模型

    env.iterator = iter(loaders[1])
    for e in range(10):
        state = env.reset()
        score = 0  # 初始化当前回合的总奖励
        total_inference_time = 0  # 初始化当前回合的总推断时间

        for time in range(300):
            # 为字典中的每个键创建一个张量，并添加批次维度
            action = agent.act(state)  # agent 做动作
            next_state, reward, done, _ = env.step(action)
            inference_time = env.neural_network.flops[action]  # 获取推断时间
            total_inference_time += inference_time  # 累加推断时间
            score += reward  # 累加奖励
            state = next_state

            if done:
                average_inference_time = total_inference_time / time  # 计算平均推断时间
                average_inference_times.append(average_inference_time)  # 记录平均推断时间
                if average_inference_time > env.constraint:
                    score -= 1000 * (average_inference_time - env.constraint)  # 如果平均推断时间超过约束，则给
                scores.append(score)  # 记录当前回合的总奖励
                print(
                    f"Test Episode: {e}, Score: {score:.3f}, Avg ACC: {env.total_hit / time * 100:.2f},Avg Time: {average_inference_time:.4f}")
                break


# 主程序
if __name__ == "__main__":
    run()
