import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

class DecoupleNet(nn.Module):
    def __init__(self, args, n_classes=10, n_speaker=1861, input_features=40, hidden_size=256):
        super(DecoupleNet, self).__init__()
        self.args = args
        self.shared_conv = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.kws_gru = nn.GRU(64, 64, batch_first=True)
        self.sv_gru = nn.GRU(64, 64, batch_first=True)
        
        self.kws_branch = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 512)
        )
        
        self.sv_branch = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 512)

        )
        self.orth_loss = None
        self.kw_head = nn.Linear(512, n_classes)

    def forward(self, x):
        x = x.squeeze(2)
        x = self.shared_conv(x)
        kws_features, _ = self.kws_gru(x.transpose(1, 2))
        sv_features, _ = self.sv_gru(x.transpose(1, 2))
        
        kws_features = kws_features.transpose(1, 2)
        sv_features = sv_features.transpose(1, 2)
        
        k_map = self.kws_branch(kws_features)
        s_map = self.sv_branch(sv_features)
        
        out_k = self.kw_head(k_map)
        
        self.orthogonality_loss()
        return out_k, out_k, k_map, s_map

    def orthogonality_loss(self):
        # Computing the orthogonality loss between two sets of weights
        # It could be enhanced by iterating over layers if needed
        w1 = self.state_dict()['kws_gru.weight_ih_l0']
        w2 = self.state_dict()['sv_gru.weight_ih_l0']
        self.orth_loss =  torch.sum(torch.abs(torch.matmul(w1.transpose(0, 1), w2)))

def compute_loss(kws_output, sv_output, kws_target, sv_target, w1, w2, orth_lambda=0.1):
    kws_loss = F.cross_entropy(kws_output, kws_target)
    sv_loss = F.cross_entropy(sv_output, sv_target)
    orth_loss = model.orthogonality_loss(w1, w2)
    total_loss = kws_loss + sv_loss + orth_lambda * orth_loss
    return total_loss

if __name__ == "__main__":
    args=Namespace()
    args.orth_loss = "no"
    args.denoise = "no"
    device = "cpu"
    
    model = TwoBranchNetwork(num_keywords=10, num_speakers=20, input_features=40, hidden_size=256, args=args)
    input_tensor = torch.randn(32, 40,1 ,101)  # Batch size 32, 100 time steps
    out_k, out_k, k_map, s_map = model(input_tensor)
    # print(kws_output.shape, sv_output.shape)
    # print(model.kws_gru.keys())
    print(model.orth_loss)
