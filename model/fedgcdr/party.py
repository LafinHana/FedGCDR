import copy
import random
import time
from tqdm import tqdm
from .model import GAT
from torch.nn.functional import sigmoid, binary_cross_entropy
import torch
from tqdm import tqdm
import numpy as np
import math

class Server:
    def __init__(self, id, d_name, num_m, total_clients, clients, evaluate_data, user_dic, args):
        self.id = id
        self.domain_name = d_name
        self.clients = clients
        self.total_clients = total_clients
        self.num_items = num_m
        self.num_users = len(clients)
        self.V = torch.randn(num_m, args.embedding_size, device=args.device)
        self.U = torch.randn(self.num_users, args.embedding_size, device=args.device)
        torch.nn.init.uniform(self.U, a=0., b=1.)
        torch.nn.init.uniform(self.V, a=0., b=1.)
        self.evaluate_data = torch.tensor(evaluate_data).to(args.device)
        self.item_gat = GAT(args, args.embedding_size, args.embedding_size, args.embedding_size)
        self.domain_attention = torch.randn(1, args.num_domain, device=args.device)
        self.user_embedding_with_attention = torch.zeros_like(self.U)
        self.item_embedding_with_attention = torch.zeros_like(self.V)
        self.lg10 = torch.Tensor([math.log(2) / math.log(i + 2) for i in range(10)]).to(args.device)
        self.lg5 = torch.Tensor([math.log(2) / math.log(i + 2) for i in range(5)]).to(args.device)
        self.user_dic = user_dic
        self.args = args
        self.mlp = None

    def mf_train(self):
        batch_num = math.ceil(self.num_users / self.args.user_batch)
        ids = copy.deepcopy(self.clients)
        np.random.shuffle(ids)
        for bt in tqdm(range(batch_num)):
            grads, p = [], []
            item_interact_table = torch.zeros(self.num_items).to(self.args.device)
            s, t = bt * self.args.user_batch, min((bt+1) * self.args.user_batch, self.args.num_users)
            batch_user = ids[s:t]
            for it in batch_user:
                if len(self.total_clients[it].train_data[self.id]) == 0:
                    continue
                map_id = self.user_dic[it][self.domain_name]
                grad, items = self.total_clients[it].train(self.id, map_id, self.U, self.V)
                grads.append(grad)
                item_interact_table[items] += 1
            item_interact_table[item_interact_table == 0] = 1
            for it, vl in enumerate(grads):
                u_grad, i_grad = vl[0], vl[1]
                map_id = self.user_dic[batch_user[it]][self.domain_name]
                self.U[map_id] -= u_grad
                self.V -= i_grad / item_interact_table.unsqueeze(1)

    def metric_at_k(self, test_predictions, k, epoch_id):
        length = int(len(test_predictions) / 100)
        test_predictions = test_predictions.reshape(length, 100)
        values, indices = torch.topk(test_predictions, k, dim=1, largest=True)
        loc = indices == 99
        hr = torch.sum(loc).item() / length
        if k == 10: ndcg = torch.sum(self.lg10 * loc).item() / length
        else: ndcg = torch.sum(self.lg5 * loc).item() / length
        return hr, ndcg

    def test(self, U, V, epoch_id):
        test_data = self.evaluate_data
        with torch.no_grad():
            test_user, test_item = test_data[:, 0], test_data[:, 1]
            test_predictions = sigmoid(torch.sum(torch.multiply(U[test_user], V[test_item]), dim=-1))
            hr_5, ndcg_5 = self.metric_at_k(test_predictions, 5, epoch_id)
            hr_10, ndcg_10 = self.metric_at_k(test_predictions, 10, epoch_id)
            return hr_5, ndcg_5, hr_10, ndcg_10

    def test_gat(self, epoch_id):
        self.item_gat.eval()
        return self.test(self.user_embedding_with_attention, self.V, epoch_id)

    def test_mf(self, epoch_id):
        return self.test(self.U, self.V, epoch_id)

    def train_mlp(self, batch):
        s, t = batch * self.args.user_batch, min((batch + 1) * self.args.user_batch, self.args.num_users)
        selected_clients = [i for i in range(s, t)]
        grads = []
        for it in selected_clients:
            grads.append(self.total_clients[it].train_mlp(self.mlp))
        p = 1 / len(self.total_clients)
        for it in grads:
            for d in range(self.args.num_domain-1):
                gd = it[d]
                for i, vl in enumerate(self.mlp[d].parameters()):
                    vl.data -= p * gd[i]

    def kt_stage(self, tf_flag=False):
        batch_num = math.ceil(self.num_users / self.args.user_batch)
        ids = copy.deepcopy(self.clients)
        np.random.shuffle(ids)
        for bt in tqdm(range(batch_num)):
            grads_model, p, grads_embedding, grads_kt = [], [], [], []
            total_item_interact_table = torch.zeros(self.num_items).to(self.args.device)
            s, t = bt * self.args.user_batch, min((bt+1) * self.args.user_batch, self.num_users)
            batch_user = ids[s:t]
            no_trans = self.args.user_batch * 1
            for i, it in enumerate(batch_user):
                if len(self.total_clients[it].train_data[self.id]) == 0:
                    continue
                if tf_flag is False or i >= no_trans:
                    pk, grad_gat, grad_emb, grad_kt = self.total_clients[it].train_gat(self.id, self.user_dic, self.item_gat, self.U, self.V)
                else:
                    pk, grad_gat, grad_emb, grad_kt = self.total_clients[it].knowledge_transfer(self.id, self.mlp, self.user_dic, self.item_gat, self.U, self.V, self.domain_attention)
                    grads_kt.append(grad_kt)
                total_items = grad_emb[3]
                total_item_interact_table[total_items] += 1
                p.append(pk)
                grads_model.append(grad_gat)
                grads_embedding.append(grad_emb)

            p = torch.Tensor(p)
            p = p / torch.sum(p)
            for i, it in enumerate(grads_model):
                if tf_flag and i < no_trans:
                    self.domain_attention.data -= p[i] * grads_kt[i][0]
                    for mid, mlp in enumerate(self.mlp):
                        for pid, para in enumerate(mlp.parameters()):
                            try:
                                para.data -= p[i] * grads_kt[i][mid+1][pid]
                            except:
                                pass
                for j, vl in enumerate(self.item_gat.parameters()):
                    vl.data -= p[i] * it[j]
            total_item_interact_table[total_item_interact_table == 0] = 1
            for grad in grads_embedding:
                uid, u_emb, u_grad, total_items, total_grads = grad[0], grad[1], grad[2], grad[3], grad[4]
                map_id = self.user_dic[uid][self.domain_name]
                self.user_embedding_with_attention[map_id] = u_emb
                self.U[map_id] -= u_grad
                self.V[total_items] -= total_grads / total_item_interact_table[total_items].unsqueeze(1)


class Client:
    def __init__(self, id, train_data, num_m, rating_mean, domain_names, args):
        self.id = id
        self.rating_mean = rating_mean
        self.train_data = [torch.tensor(train_data[i], device=args.device) for i in range(args.num_domain)]
        self.items = train_data
        self.gat = None
        self.knowledge = [[] for _ in range(args.num_domain)]
        self.num_items = num_m
        self.unselected = []
        self.mlp = []
        self.args = args
        self.delta = torch.tensor(args.delta, device=args.device)
        self.sensitivity = torch.sqrt(torch.tensor(1, device=args.device))
        self.domain_names = domain_names

    def reset(self, input):
        output = torch.clone(input).detach()
        output.requires_grad = True
        output.grad = torch.zeros_like(output)
        return output

    @staticmethod
    def sample_negative(data, num):
        neg = torch.randint(0, num, (4 * len(data), 1), device=data.device, dtype=torch.int64).squeeze()
        rating = torch.cat((torch.ones(len(data), device=data.device),
                            torch.zeros(len(neg), device=data.device)), dim=0)
        neg = torch.cat((data, neg), dim=0)
        return neg, rating

    def train(self, domain_id, map_id, user_embedding, item_embedding):
        domain_items, domain_ratings = self.sample_negative(self.train_data[domain_id], self.num_items[domain_id])
        item_emb = self.reset(item_embedding)
        user_emb = self.reset(user_embedding[map_id])
        optimizer = torch.optim.Adam([user_emb, item_emb], lr=self.args.lr_mf)
        for _ in range(self.args.local_epoch):
            optimizer.zero_grad()
            predict = torch.sum(torch.multiply(user_emb, item_emb[domain_items]), dim=1)
            predict = sigmoid(predict)
            loss = binary_cross_entropy(predict, domain_ratings)
            loss.backward()
            optimizer.step()
        grads = [user_embedding[map_id].detach() - user_emb.detach(), item_embedding.detach() - item_emb.detach()]
        return grads, domain_items

    def train_gat(self, domain_id, user_dic, model_item, global_user_embedding, global_item_embedding, transfer=False, a=None ,transfer_vec=None):

        grads_gat, grad_emb, grad_kt, temp_vec = [], [], [], [0 for _ in range(self.args.num_domain)]
        length = len(self.items[domain_id])
        self.gat = copy.deepcopy(model_item)
        user_embedding = self.reset(global_user_embedding[user_dic[self.id][self.domain_names[domain_id]]])
        item_embedding = self.reset(global_item_embedding)
        paras = [user_embedding, item_embedding] + [para for para in self.gat.parameters()]
        local_a = a
        if transfer:
            mlps = copy.deepcopy(self.mlp)
            for mlp in mlps:
                paras += [para for para in mlp.parameters()]
            local_a = self.reset(a)
        optimizer = torch.optim.Adam(paras, lr=self.args.lr_gat)
        total_item, ratings = self.sample_negative(self.train_data[domain_id], self.num_items[domain_id])
        for epoch in range(self.args.local_epoch):
            optimizer.zero_grad()
            if transfer:
                for i in range(self.args.num_domain):
                    temp_vec[i] = mlps[i](transfer_vec[i])
            h_i, intermediate_emb, ls, lm = self.gat(
                torch.cat((user_embedding.reshape(1, self.args.embedding_size), item_embedding[self.items[domain_id]])),
                transfer, local_a, temp_vec)
            user_emb = h_i[0]
            h_i = item_embedding[total_item]
            predict = sigmoid(torch.sum(torch.multiply(user_emb, h_i), dim=1))
            loss = binary_cross_entropy(predict, ratings) + ls + lm
            loss.backward()
            optimizer.step()

        local_para = [para.data for para in self.gat.parameters()]
        global_para = [para.data for para in model_item.parameters()]
        for i in range(len(local_para)):
            grads_gat.append(global_para[i] - local_para[i])
        with torch.no_grad():
            user_emb, self.knowledge[domain_id], ls, lm = self.gat(
                torch.cat((user_embedding.reshape(1, self.args.embedding_size), item_embedding[self.items[domain_id]])),
                transfer, local_a, transfer_vec)
        grad_emb.append(self.id)
        grad_emb.append(user_emb[0].detach())
        grad_emb.append(user_embedding.detach())
        grad_emb.append(total_item)
        grad_emb.append(global_item_embedding[grad_emb[-1]].detach() - item_embedding[grad_emb[-1]].detach())
        if transfer:
            grad_kt.append(a.detach() - local_a.detach())
            for i in range(self.args.num_domain):
                local_para = [para.data for para in mlps[i].parameters()]
                global_para = [para.data for para in self.mlp[i].parameters()]
                para_grad = []
                for pid in range(len(local_para)):
                    para_grad.append(global_para[pid] - local_para[pid])
                grad_kt.append(para_grad)
        return length, grads_gat, grad_emb, grad_kt

    @staticmethod
    def l2_clip(x, s):
        norm = torch.norm(x)
        if norm > s:
            return s * (x / norm)
        else:
            return x

    def knowledge_transfer(self,domain_id, mlps, user_dic, item_gat, user_embedding, item_embedding, a):
        transfer_vec = []
        self.mlp = mlps
        std = self.sensitivity * torch.sqrt(2 * torch.log(1.25 / self.delta)) * 1 / self.args.eps
        for j in range(self.args.num_domain):
            if j == domain_id:
                transfer_vec.append(torch.zeros(self.args.embedding_size, device=self.args.device))
            else:
                if len(self.knowledge[j]) == 0:
                    temp_vec = torch.zeros(self.args.embedding_size, device=self.args.device)
                else:
                    temp_vec = Client.l2_clip(torch.tensor(self.knowledge[j][0], device=self.args.device), self.sensitivity)
                noise = torch.normal(mean=0, std=std, size = (1, self.args.embedding_size)).to(self.args.device).squeeze()
                if self.args.dp:
                    transfer_vec.append(temp_vec + noise)
                else:
                    transfer_vec.append(temp_vec)

        return self.train_gat(domain_id, user_dic, item_gat , user_embedding, item_embedding,True, a, transfer_vec)

