from __future__ import annotations

import numpy as np


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install torch on the GPU server before running this script."
        ) from exc
    return torch


def build_normalized_adj(data, device):
    torch = require_torch()
    rows = data.train_pairs[:, 0]
    cols = data.train_pairs[:, 1] + data.n_users
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    values = np.ones(len(all_rows), dtype=np.float32)
    degrees = np.bincount(all_rows, minlength=data.n_users + data.n_items).astype(np.float32)
    norm = values / np.sqrt(degrees[all_rows] * degrees[all_cols] + 1.0e-12)
    indices = torch.tensor(np.vstack([all_rows, all_cols]), dtype=torch.long, device=device)
    values_t = torch.tensor(norm, dtype=torch.float32, device=device)
    shape = (data.n_users + data.n_items, data.n_users + data.n_items)
    return torch.sparse_coo_tensor(indices, values_t, shape, device=device).coalesce()


class BPRMF:
    def __init__(self, n_users, n_items, dim, device):
        torch = require_torch()
        self.torch = torch
        self.model = _BPRMFModule(n_users, n_items, dim).to(device)
        self.device = device

    def parameters(self):
        return self.model.parameters()

    def loss(self, users, pos_items, neg_items, weight_decay):
        torch = self.torch
        pos_scores = self.model(users, pos_items)
        neg_scores = self.model(users, neg_items)
        bpr = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
        reg = (
            self.model.user_emb(users).pow(2).sum()
            + self.model.item_emb(pos_items).pow(2).sum()
            + self.model.item_emb(neg_items).pow(2).sum()
        ) / users.shape[0]
        return bpr + weight_decay * reg

    def score_all_items(self, user_ids):
        torch = self.torch
        with torch.no_grad():
            users = torch.as_tensor(user_ids, dtype=torch.long, device=self.device)
            user_emb = self.model.user_emb(users)
            item_emb = self.model.item_emb.weight
            return (user_emb @ item_emb.T).detach().cpu().numpy()


class LightGCN:
    def __init__(self, data, dim, layers, device):
        torch = require_torch()
        self.torch = torch
        self.model = _LightGCNModule(data.n_users, data.n_items, dim, layers).to(device)
        self.adj = build_normalized_adj(data, device)
        self.device = device

    def parameters(self):
        return self.model.parameters()

    def loss(self, users, pos_items, neg_items, weight_decay):
        torch = self.torch
        user_emb, item_emb = self.model.propagate(self.adj)
        pos_scores = (user_emb[users] * item_emb[pos_items]).sum(dim=1)
        neg_scores = (user_emb[users] * item_emb[neg_items]).sum(dim=1)
        bpr = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
        reg = (
            self.model.user_emb(users).pow(2).sum()
            + self.model.item_emb(pos_items).pow(2).sum()
            + self.model.item_emb(neg_items).pow(2).sum()
        ) / users.shape[0]
        return bpr + weight_decay * reg

    def score_all_items(self, user_ids):
        torch = self.torch
        with torch.no_grad():
            user_emb, item_emb = self.model.propagate(self.adj)
            users = torch.as_tensor(user_ids, dtype=torch.long, device=self.device)
            return (user_emb[users] @ item_emb.T).detach().cpu().numpy()


class _BPRMFModule:
    def __new__(cls, n_users, n_items, dim):
        torch = require_torch()

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.user_emb = torch.nn.Embedding(n_users, dim)
                self.item_emb = torch.nn.Embedding(n_items, dim)
                torch.nn.init.xavier_uniform_(self.user_emb.weight)
                torch.nn.init.xavier_uniform_(self.item_emb.weight)

            def forward(self, users, items):
                return (self.user_emb(users) * self.item_emb(items)).sum(dim=1)

        return Module()


class _LightGCNModule:
    def __new__(cls, n_users, n_items, dim, layers):
        torch = require_torch()

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.user_emb = torch.nn.Embedding(n_users, dim)
                self.item_emb = torch.nn.Embedding(n_items, dim)
                self.layers = layers
                torch.nn.init.xavier_uniform_(self.user_emb.weight)
                torch.nn.init.xavier_uniform_(self.item_emb.weight)

            def propagate(self, adj):
                emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
                outputs = [emb]
                for _ in range(self.layers):
                    emb = torch.sparse.mm(adj, emb)
                    outputs.append(emb)
                final = torch.stack(outputs, dim=0).mean(dim=0)
                return final[:n_users], final[n_users:]

        return Module()
