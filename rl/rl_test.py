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
import brevitas.nn as qnn
from brevitas.quant import Int32Bias

# 定义能量消耗计算函数
def calculate_energy(flops, exit_point):
    # 假设每个FLOP消耗的能量是一个常数
    energy_per_flop = 0.9# 可以根据实际情况调整
    return flops[exit_point] * energy_per_flop



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
                                    "avg_latency": spaces.Box(low=float(0), high=float(1),shape=(np.array([1])))
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


def cal_flops():
    input = torch.randn(1, 40, 1, 101)
    # 使用thop计算FLOPs和参数数量
    flops, params = profile(ee_model, inputs=(input,))
    print(f"FLOPs: {flops}") #  634800.0  1503888
    print(f"Params: {params}")

    input2 = {"feature": input,  # 或者其他适合你应用的初始化方式
                "avg_latency": torch.Tensor([0.5])} #
    flops, params = profile(DQN(feat_shape, action_size), inputs=(input2,))
    print(f"FLOPs: {flops}")        # 279520
    print(f"Params: {params}")      # 5862


# 主程序
if __name__ == "__main__":
    root_dir = "../dataset/lege/"
    word_list = ['上升', '下降', '乐歌', '停止', '升高', '坐', '复位', '小乐', '站', '降低']
    speaker_list = []
    loaders = sd.kws_loaders(root_dir, word_list, speaker_list)
    # ee_model = QuantizedTCResNet8(1, 40, 10)
    ee_model = TCResNet8(1, 40, 10)
    ee_model.load("../saved_model/lege_ee_float_96.333.pt")
    ee_model.eval()

    env = EarlyExitEnv(ee_model, loaders[1], constraint=0.8)
    action_size = env.action_space.n    # 2, 从哪个exit出
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
            action = agent.act(state)           # agent 做动作
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
                    score -= 1000*(average_inference_time - env.constraint)  # 如果平均推断时间超过约束，则给予负奖励
                scores.append(score)  # 记录当前回合的总奖励
                print(
                    f"Episode: {e}/{episodes}, Score: {score:.3f}, Avg ACC: {env.total_hit/time*100:.2f},Avg Time: {average_inference_time:.4f}, Epsilon: {agent.epsilon:.4}")
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

