from argparse import Namespace
import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# 定义SNN模型
class SNNKeywordSpotting(nn.Module):
    def __init__(self, num_hidden=128, n_class=10, num_steps=100, beta=0.95):
        super(SNNKeywordSpotting, self).__init__()
        
        # 将超参数保存为模型属性
        self.num_hidden = num_hidden
        self.n_class = n_class
        self.num_steps = num_steps
        self.beta = beta
        
        # 网络结构：输入形状为 [batch_size, 40, 1, 101]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1)  # 卷积层，输入通道数1，输出通道数32
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.fc1 = nn.Linear(32 * 20 * 50, num_hidden)  # 展平后接入全连接层
        self.lif1 = snn.Leaky(beta=beta)  # 第一层LIF神经元
        self.fc2 = nn.Linear(num_hidden, n_class)  # 第二层全连接层，输出类别数
        self.lif2 = snn.Leaky(beta=beta)  # 第二层LIF神经元
    
    def forward(self, x):
        # 输入x的形状为 [batch_size, 40, 1, 101]
        x = x.view(x.size(0), 1, 40, 101)  # 将输入reshape为 [batch_size, 1, 40, 101] 以适应卷积层
        x = self.pool(torch.relu(self.conv1(x)))  # 通过卷积层和池化层
        
        # 将卷积层输出展平
        x = x.view(x.size(0), -1)  # 形状为 [batch_size, 32 * 20 * 50]
        
        # 初始化膜电位
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = []  # 记录脉冲输出
        mem2_rec = []  # 记录膜电位
        
        # 时间步数迭代
        for step in range(self.num_steps):
            cur1 = self.fc1(x)  # 全连接层
            spk1, mem1 = self.lif1(cur1, mem1)  # LIF神经元
            cur2 = self.fc2(spk1)  # 第二个全连接层
            spk2, mem2 = self.lif2(cur2, mem2)  # 第二个LIF神经元
            spk2_rec.append(spk2)  # 记录脉冲
            mem2_rec.append(mem2)  # 记录膜电位
        
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)



# Spike Function 使用继承 autograd.Function
class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vth):
        ctx.save_for_backward(input)
        return (input > vth).float()  # 产生脉冲

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0  # 使用阈值来计算梯度
        return grad_input, None


class SpikGRUCell(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, vth=1.0):
        super(SpikGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vth = vth

        # 定义 GRU 参数
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size)

        # 定义自定义脉冲函数
        self.spike_fn = SpikeFunction.apply

    def forward(self, x_t, h_t):
        # 计算更新门 z
        z_t = torch.sigmoid(self.W_z(x_t) + self.U_z(h_t))
        # 输入调制 i
        i_t = self.W_i(x_t) + self.U_i(h_t)

        # 膜电位更新 v
        v_t = z_t * h_t + (1 - z_t) * i_t

        # 生成脉冲
        spike = self.spike_fn(v_t - self.vth, self.vth)

        # 更新隐藏状态
        h_t_new = (1 - spike) * v_t + spike * v_t
        return h_t_new, spike


class SpikGRU(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, output_size=10):
        super(SpikGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # 使用多层 GRU
        self.cells = nn.ModuleList([SpikGRUCell(input_size, hidden_size) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x 的输入形状为 [time_steps, batch_size, 40, 1, 101]
        x = x.squeeze(3)  # 去掉单通道维度，形状变为 [time_steps, batch_size, 40, 101]
        x = x.transpose(2, 3)  # 调整为 [time_steps, batch_size, 101, 40]，符合 RNN 输入

        time_steps, batch_size, num_steps, _ = x.size()  # 获取时间步长和其他维度
        h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        spikes = []

        # 遍历每个时间步
        for t in range(time_steps):
            for step in range(num_steps):  # 遍历每个特征时间步
                x_t = x[t, :, step, :]  # 当前时间步的数据，形状 [batch_size, 40]
                for i, cell in enumerate(self.cells):
                    h[i], s_t = cell(x_t, h[i])  # 更新每一层的状态
                spikes.append(s_t)

        # 堆叠所有时间步的输出，并在时间维度上求和
        spike_stack = torch.stack(spikes, dim=1)  # [batch_size, num_steps * time_steps, hidden_size]
        spike_sum = torch.sum(spike_stack, dim=1)  # 在时间步上求和
        output = self.fc(spike_sum)  # 最终分类输出
        return output
if __name__ == "__main__":
    x = torch.rand(7, 40, 1, 101)
    # TCResNet8 test  model =  StarNet(num_classes=10)
    # model =  SNNKeywordSpotting()
    model = SpikGRU()
    result_tcresnet8 = model(x)
    print(len(result_tcresnet8), result_tcresnet8[0].shape)


    # args=Namespace()
    # args.orth_loss = "no"
    # args.denoise = "no"
    # device = "cpu"
    # model = StarNet()
    # macs, params = thop.profile(model,inputs=(x,))
    # print(f"SNN  macs {macs}, params {params}")