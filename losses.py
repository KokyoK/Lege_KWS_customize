import torch
torch.manual_seed(42)
import math
import torch.nn as nn

class ConfCrLoss(nn.Module):
    def __init__(self):
        super(ConfCrLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_loss_each = nn.CrossEntropyLoss(reduction='none')
        self.loss_type = "hard"

    def forward(self,conti_outs,  label, exit_indices,non_exit_indices,epoch,e_index,a):
        [prev_predicted, predicted, next_predicted] = conti_outs


        ce = self.ce_loss(predicted, label)
        return ce
