import torch.nn as nn
import torch

class GATLayer(nn.Module):
    def __init__(self, in_feature, out_feature, alpha):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.A = nn.Parameter(torch.empty(size=(2 * out_feature, 1)))
        nn.init.xavier_uniform_(self.A.data, nn.init.calculate_gain('relu'))
        self.alpha = alpha

    def forward(self, input, adj):
        h = input
        h1 = torch.matmul(h, self.A[self.out_feature:, :])
        h2 = torch.matmul(h, self.A[:self.out_feature, :])
        e = h1 + h2.T
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=-1)
        ah = torch.matmul(attention, h)
        return ah


class GAT(nn.Module):
    def __init__(self, args, in_feature, hid_feature=16, out_feature=16, alpha=0.1, dropout=0):
        super().__init__()
        self.in_feature = in_feature
        self.hid_feature = hid_feature
        self.out_feature = out_feature
        self.drop = nn.Dropout(p=dropout)
        self.in2hidden = GATLayer(in_feature, hid_feature, alpha).to(args.device)
        self.hidden2out = GATLayer(hid_feature, out_feature, alpha).to(args.device)

    @staticmethod
    def compute_ls(f_t, f_s):
        total_sim = 0
        for fs in f_s:
            with torch.no_grad():
                sim = torch.cosine_similarity(fs, f_t, dim=0)
                total_sim += sim
        F_s = 0
        for fs in f_s:
            with torch.no_grad():
                sim = torch.cosine_similarity(fs, f_t, dim=0)
            F_s += sim * fs / total_sim
        loss = torch.norm(f_t - F_s) ** 2
        return loss

    @staticmethod
    def compute_lm(f_t, f_s):
        loss = 0
        for fs in f_s:
            loss += torch.nn.functional.mse_loss(fs, f_t)
        return loss

    def forward(self, x, is_transfer_stage=False, domain_attention=None, transfer_vec=None):
        ls, lm = 0, 0
        alpha, beta = 0.01, 0.01
        intermediate_embedding = []
        adj = torch.eye(len(x), device=x.device)
        adj[:, 0] = 1.
        adj[0, :] = 1.
        x = self.in2hidden(x, adj)
        intermediate_embedding.append(x[0].data)
        if is_transfer_stage:
            ls = alpha / 2 * self.compute_ls(x[0], transfer_vec)
            lm = beta / 2 * self.compute_lm(x[0], transfer_vec)
            transfer_vec = torch.stack(transfer_vec)
            x = torch.cat((x, transfer_vec))
            adj = torch.eye(len(x), device=x.device)
            adj[:, 0] = 1.
            adj[0, :] = 1.
        x = self.hidden2out(x, adj)
        return x, intermediate_embedding, ls, lm


class MLP(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.L1 = nn.Linear(in_feature, 2*in_feature)
        self.L2 = nn.Linear(2*in_feature, int(in_feature/2))
        self.L3 = nn.Linear(int(in_feature/2), in_feature)
        self.f = nn.Tanh()

    def forward(self, x):
        x = self.f(self.L1(x))
        x = self.f(self.L2(x))
        x = self.f(self.L3(x))
        return x