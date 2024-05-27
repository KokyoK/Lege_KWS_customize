# import torch

# # Step 1: 加载模型状态字典
# model_path = 'saved_model/google_noisy/star_o_u_23_kwsacc_81.58_idloss_0.3149'
# state_dict = torch.load(model_path)

# # Step 2: 修改键名
# # 只有键名以'network.w_'开头的才替换为'network.orth_block.w_'
# new_state_dict = {}
# for key in state_dict:
#     if key.startswith('network.w_'):
#         new_key = 'network.orth_block.' + key[len('network.'):]  # 替换部分开始于'network.'
#     else:
#         new_key = key
#     new_state_dict[new_key] = state_dict[key]

# # Step 3: 保存修改后的模型状态字典
# new_model_path = 'saved_model/google_noisy/star_o_u_23_kwsacc_81.58_idloss_0.3149'
# torch.save(new_state_dict, new_model_path)
# print("Modified model saved to", new_model_path)

import pandas as pd

# 读取CSV文件
file_path = 'dataset/google_noisy/split_align/test.csv'
data = pd.read_csv(file_path)

# 统计不重复的speaker ID数量
unique_speakers = data['path'].nunique()

print(f"Number of unique speakers: {unique_speakers}")
