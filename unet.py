import torch
import torch.nn as nn
import torch.nn.functional as F
# import speech_dataset as sd
# import noisy_dataset as nsd 
# from timm.models.layers import DropPath, trunc_normal_
# from timm.models.registry import register_model
import thop
import time
# from ptflops import get_model_complexity_info
import copy
from argparse import Namespace

from models.TCResNets import S1_Block,S2_Block
torch.manual_seed(42)


# class OrthBlock(nn.Module):
#     def __init__(self, feature_dim, num_heads=1):
#         super(OrthBlock, self).__init__()
#         self.feature_dim = feature_dim
#         self.num_heads = num_heads
        
#         self.attention_ks = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=num_heads, batch_first=True)
#         self.attention_sk = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=num_heads, batch_first=True)
        
#         self.linear_kk = nn.Linear(self.feature_dim, self.feature_dim)
#         self.linear_ks = nn.Linear(self.feature_dim, self.feature_dim)
#         self.linear_ss = nn.Linear(self.feature_dim, self.feature_dim)
#         self.linear_sk = nn.Linear(self.feature_dim, self.feature_dim)

#     def forward(self, k_map, s_map):
#         batch_size, feature_dim, _, seq_length = k_map.size()
        
#         # Reshape to (batch_size, seq_length, feature_dim) to use with nn.MultiheadAttention
#         k_map_seq = k_map.squeeze(2).permute(0, 2, 1)  # Shape: (batch_size, seq_length, feature_dim)
#         s_map_seq = s_map.squeeze(2).permute(0, 2, 1)  # Shape: (batch_size, seq_length, feature_dim)

#         # Perform cross attention
#         k_to_s_output, _ = self.attention_ks(query=s_map_seq, key=k_map_seq, value=k_map_seq)  # s_map queries k_map
#         s_to_k_output, _ = self.attention_sk(query=k_map_seq, key=s_map_seq, value=s_map_seq)  # k_map queries s_map

#         # Linear transformations
#         kk_transformed = self.linear_kk(k_map_seq)
#         ks_transformed = self.linear_ks(k_to_s_output)
#         ss_transformed = self.linear_ss(s_map_seq)
#         sk_transformed = self.linear_sk(s_to_k_output)


#         # Combine features
#         k_final = kk_transformed + sk_transformed  # Shape: (batch_size, seq_length, feature_dim)
#         s_final = ss_transformed + ks_transformed  # Shape: (batch_size, seq_length, feature_dim)

#         # Reshape back to original shape
#         k_final = k_final.permute(0, 2, 1).unsqueeze(2)  # Shape: (batch_size, feature_dim, 1, seq_length)
#         s_final = s_final.permute(0, 2, 1).unsqueeze(2)  # Shape: (batch_size, feature_dim, 1, seq_length)

#         return k_final, s_final
import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(OrthogonalMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.o_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.xavier_uniform_(self.o_proj_weight)

    def orthogonalize(self, weight1, weight2):
        with torch.no_grad():
            weight1_reshaped = weight1.view(weight1.size(0), -1)
            weight2_reshaped = weight2.view(weight2.size(0), -1)
            weight1_proj = weight1_reshaped - torch.mm(weight1_reshaped, torch.mm(weight2_reshaped.t(), weight2_reshaped))
            weight2_proj = weight2_reshaped - torch.mm(weight2_reshaped, torch.mm(weight1_reshaped.t(), weight1_reshaped))
            weight1.copy_(weight1_proj.view_as(weight1))
            weight2.copy_(weight2_proj.view_as(weight2))

    def forward(self, query, key, value):
        # Ensuring orthogonality of query and key projection weights
        self.orthogonalize(self.q_proj_weight, self.k_proj_weight)

        batch_size = query.size(0)

        # Linear projections
        q = F.linear(query, self.q_proj_weight)
        k = F.linear(key, self.k_proj_weight)
        v = F.linear(value, self.v_proj_weight)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        attn_output = F.linear(attn_output, self.o_proj_weight)

        return attn_output

class OrthBlock(nn.Module):
    def __init__(self, feature_dim, num_heads=1):
        super(OrthBlock, self).__init__()
        self.feature_dim = feature_dim

        self.attention_kk = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=num_heads, batch_first=True)
        self.attention_ks = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=num_heads, batch_first=True)
        self.attention_ss = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=num_heads, batch_first=True)
        self.attention_sk = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=num_heads, batch_first=True)
        print(self.attention_kk)

    def orthogonalize(self, weight1, weight2):
        with torch.no_grad():
            weight1_reshaped = weight1.view(weight1.size(0), -1)
            weight2_reshaped = weight2.view(weight2.size(0), -1)
            weight1_proj = weight1_reshaped - torch.mm(weight1_reshaped, torch.mm(weight2_reshaped.t(), weight2_reshaped))
            weight2_proj = weight2_reshaped - torch.mm(weight2_reshaped, torch.mm(weight1_reshaped.t(), weight1_reshaped))
            weight1.copy_(weight1_proj.view_as(weight1))
            weight2.copy_(weight2_proj.view_as(weight2))

    def forward(self, k_map, s_map):
        batch_size, feature_dim, _, seq_length = k_map.size()

        # Reshape to (batch_size, seq_length, feature_dim) to use with nn.MultiheadAttention
        k_map_seq = k_map.squeeze(2).permute(0, 2, 1)  # Shape: (batch_size, seq_length, feature_dim)
        s_map_seq = s_map.squeeze(2).permute(0, 2, 1)  # Shape: (batch_size, seq_length, feature_dim)

        # Ensure orthogonality of query and key projection weights for attention_kk and attention_ks
        # self.orthogonalize(self.attention_kk.in_proj_weight[:self.feature_dim], self.attention_ks.in_proj_weight[:self.feature_dim])
        # self.orthogonalize(self.attention_ss.in_proj_weight[:self.feature_dim], self.attention_sk.in_proj_weight[:self.feature_dim])

        # Perform attention
        kk_output, _ = self.attention_kk(k_map_seq, k_map_seq, k_map_seq)  # Self-attention on k_map
        ks_output, _ = self.attention_ks(k_map_seq, s_map_seq, s_map_seq)  # Cross-attention: k_map queries s_map
        ss_output, _ = self.attention_ss(s_map_seq, s_map_seq, s_map_seq)  # Self-attention on s_map
        sk_output, _ = self.attention_sk(s_map_seq, k_map_seq, k_map_seq)  # Cross-attention: s_map queries k_map

        # Combine features
        k_combined = kk_output + sk_output  # Shape: (batch_size, seq_length, feature_dim)
        s_combined = ss_output + ks_output  # Shape: (batch_size, seq_length, feature_dim)

        # Reshape back to original shape
        k_final = k_map + k_combined.permute(0, 2, 1).unsqueeze(2)  # Shape: (batch_size, feature_dim, 1, seq_length)
        s_final = s_map + s_combined.permute(0, 2, 1).unsqueeze(2)  # Shape: (batch_size, feature_dim, 1, seq_length)

        return k_final, s_final



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 编码器
        self.conv1 = nn.Conv2d(40, 24, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(24, 48, kernel_size=(1, 3), padding=(0, 1))
        self.conv3 = nn.Conv2d(48, 96, kernel_size=(1, 3), padding=(0, 1))
        self.pool = nn.MaxPool2d((1, 2))
        self.B1 = nn.BatchNorm2d(24)
        self.B2 = nn.BatchNorm2d(48)

        self.B3 = nn.BatchNorm2d(96)
        self.B4 = nn.BatchNorm2d(48)
        self.B5 = nn.BatchNorm2d(24)
        self.B6 = nn.BatchNorm2d(24)


        # 解码器
        self.upconv1 = nn.ConvTranspose2d(96, 48, kernel_size=(1, 2), stride=(1, 2))
        self.conv4 = nn.Conv2d(96, 48, kernel_size=(1, 3), padding=(0, 1))
        self.upconv2 = nn.ConvTranspose2d(48, 24, kernel_size=(1, 2), stride=(1, 2), output_padding=(0, 1))
        self.conv5 = nn.Conv2d(48, 24, kernel_size=(1, 3), padding=(0, 1))
        self.conv6 = nn.Conv2d(24, 40, kernel_size=(1, 3), padding=(0, 1))
        self.fc_mu = nn.Linear(25*96, 128)
        self.fc_var = nn.Linear(25*96, 128)
        self.decoder_input = nn.Linear(128, 25*96)

        # 激活函数
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        # 编码器
        c1 = self.relu(self.conv1(x))
        p1 = self.B1(self.pool(c1))
        c2 = self.relu(self.conv2(p1))
        p2 = self.B2(self.pool(c2))
        c3 = self.relu(self.B3(self.conv3(p2)))

        result = torch.flatten(c3, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        z = self.reparameterize(mu, log_var)
        z = self.decoder_input(z).reshape(c3.shape[0], c3.shape[1], c3.shape[2], c3.shape[3])


        # 解码器
        up1 = self.relu(self.B4(self.upconv1(z)))
        merge1 = torch.cat([up1, c2], dim=1)
        c4 = self.relu(self.conv4(merge1))
        up2 = self.relu(self.B5(self.upconv2(c4)))
        merge2 = torch.cat([up2, c1], dim=1)
        c5 = self.relu(self.conv5(merge2))
        c6 = self.tanh(self.conv6(c5))

        return c6, mu, log_var

class ConvBN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups)
        self.bn = nn.BatchNorm2d(out_planes) if with_bn else nn.Identity()
        if with_bn:
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=2, drop_path=0):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, [1, 7], 1, (0, 3), groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.g(x)
        x = input + self.drop_path(x)
        return x


class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[1, 1, ], mlp_ratio=4, drop_path_rate=0, num_classes=10,n_speaker=1841, args=None):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.in_channel = 32
        self.stem = nn.Sequential(
            ConvBN(40, self.in_channel, kernel_size=[1, 3], stride=1, padding=[0, 1]),
            nn.ReLU6(),
            # S1_Block(32)
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        # for i_layer in range(len(depths)):
        #     embed_dim = base_dim * 2 ** i_layer
        #     down_sampler = ConvBN(self.in_channel, embed_dim, [1, 3], 2, [0, 1])
        #     self.in_channel = embed_dim
        #     blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
        #     cur += depths[i_layer]
        #     self.stages.append(nn.Sequential(down_sampler, *blocks))
        
        embed_dim = base_dim
        down_sampler = ConvBN(self.in_channel, embed_dim, [1, 3], 2, [0, 1])
        self.in_channel = embed_dim
        blocks = [Block(self.in_channel, mlp_ratio, dpr[0])]
        cur += depths[0]
        self.stages.append(nn.Sequential(down_sampler, *blocks))

        embed_dim = base_dim * 2
        down_sampler = ConvBN(self.in_channel, embed_dim, [1, 3], 2, [0, 1])
        down_sampler_s = ConvBN(self.in_channel, embed_dim, [1, 3], 2, [0, 1])
        self.in_channel = embed_dim  # 更新 self.in_channel
        blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[1])]
        blocks_s = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[1])]
        # cur += depths[1]  # 更新 cur
        self.stages.append(nn.Sequential(down_sampler, *blocks))
       
        self.s_block = nn.Sequential(down_sampler_s, *blocks_s)
        self.k_block = nn.Sequential(down_sampler, *blocks)
        self.k_block_tc = nn.Sequential(S1_Block(32),S2_Block(32,48))

        
        # self.s_block = nn.Sequential(down_sampler, *blocks)
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.head_k = nn.Linear(self.in_channel, num_classes)
        # self.head_s = nn.Linear(self.in_channel, n_speaker)
        
        self.orth_block = OrthBlock(feature_dim=64)
        # self.apply(self._init_weights)
        # self.k_attn = nn.MultiheadAttention(embed_dim=13, num_heads=1)
        # self.s_attn = nn.MultiheadAttention(embed_dim=13, num_heads=1)
        
    def forward(self, x):
        x = self.stem(x)
        # for i in range(len(self.stages)-1):
        stage = self.stages[0]
        

        # if self.args.att == "yes":
        #     share_map_T = x.squeeze(2).permute(1, 0, 2)
        #     attn_k, attn_k_weights = self.k_attn(share_map_T, share_map_T, share_map_T)
        #     attn_s, attn_s_weights = self.s_attn(share_map_T, share_map_T, share_map_T)
            
        #     attn_k = attn_k.squeeze(-1).permute(1,0,2) # 移除最后一个维度
        #     attn_s = attn_s.squeeze(-1).permute(1,0,2) # 同理
            
        #     k_map_T = x.squeeze(2).permute(1, 0, 2)    
        #     s_map_T = x.squeeze(2).permute(1, 0, 2)    
            
        #     attn_k, attn_k_weights = self.k_attn(k_map_T, k_map_T, k_map_T)
        #     attn_s, attn_s_weights = self.s_attn(s_map_T, s_map_T, s_map_T)
            
        #     k_map = attn_k.permute(1, 0, 2).unsqueeze(2)
        #     s_map = attn_s.permute(1, 0, 2).unsqueeze(2)   
        # print(x.shape)
        x = stage(x)
        # x = stage(x)

        
        k_map = self.k_block_tc(x)
        k_map = self.k_block(x)
        # s_map = stage(x)
        s_map = self.s_block(x)




        if self.args.orth_loss == "yes":     
            k_map, s_map = self.orth_block(k_map, s_map)
            
        k_map = self.avgpool(self.norm(k_map))
        s_map = self.avgpool(self.norm(s_map))
        
        k_map = k_map.squeeze(2,3)
        s_map = s_map.squeeze(2,3)
        # print(k_map.shape)
        out_k = self.head_k(k_map)
        # out_s = self.head_s(s_map)
        return out_k, out_k, k_map, s_map
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# You can adjust the base_dim


    
if __name__ == '__main__':
    args=Namespace()
    args.orth_loss = "yes"
    args.denoise = "no"
    


    unet = UNet()
    starnet = StarNet(args=args)
    input_tensor = torch.randn(7, 40, 1, 101)  # batch_size=4
    clean_tensor = torch.randn(2, 40, 1, 101) 

#     output_tensor = starnet(input_tensor)
#     # print(output_tensor.shape)

#     flops, params = thop.profile(unet,inputs=(input_tensor,))
#     print(f"U NET flops {flops}, params {params}")

    
    output_tensor = starnet(input_tensor)
    # print(output_tensor)
#     macs, params = thop.profile(starnet,inputs=(input_tensor,))
#     # start = time.time()
#     # for i in range(1000):
#     #     out = starnet(input_tensor)
#     # end = time.time()
#     # print("Running time of start net: ", (end-start)) # ms
#     print(f"STAR NET macs {macs}, params {params}")
    

#     # flops, params = get_model_complexity_info(starnet, (40,1,101), as_strings=True, print_per_layer_stat=True)
#     # print('flops: ', flops, 'params: ', params)

