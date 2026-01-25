# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class LearnableGegenbauer(nn.Module):

    def __init__(self, lam, max_degree, eps=1e-12, g=5.0):
        super().__init__()
        assert max_degree >= 2, "max_degree should be at least 2."
        self.lam = float(lam)
        self.eps = float(eps)
        self.K = int(max_degree)
        self.g = float(g)
        self.register_buffer("C_scaler", self._Ck1_norms(self.K, self.lam))
        init_a = torch.zeros(self.K, dtype=torch.float32)
        self.a = nn.Parameter(init_a)

    @staticmethod
    def _Ck1_norms(K, lam):
        ks  = torch.arange(K, dtype=torch.float32)
        lam = torch.tensor(lam, dtype=torch.float32)
        numer = torch.lgamma(ks + 2.0 * lam)
        demon = torch.lgamma(2.0 * lam) + torch.lgamma(ks + 1.0)
        return torch.exp(numer - demon)

    @staticmethod
    def _gegenbauer_norms(K, lam):
        ks  = torch.arange(K, dtype=torch.float32)
        lam = torch.as_tensor(lam, dtype=torch.float32)
        lgh = (torch.log(torch.tensor(torch.pi, dtype=torch.float64))
               + (1.0 - 2.0 * lam) *
               torch.log(torch.tensor(2.0, dtype=torch.float64))
               + torch.lgamma(ks + 2.0 * lam)
               - torch.lgamma(ks + 1.0)
               - torch.log(ks + lam)
               - 2.0 * torch.lgamma(lam))
        return torch.exp(0.5 * lgh)

    def _degree_weights(self):
        a = F.softplus(self.a)
        return a / (a.sum() + self.eps)
    
    def entropy(self):
        w = self._degree_weights()
        return - (w * torch.log(w + self.eps)).sum()

    def forward(self, x):
        w = self._degree_weights()
        w = w / self.C_scaler
        lam = self.lam

        C0 = torch.ones_like(x, device=x.device)
        C1 = (2.0 * lam) * x

        out = w[0] * C0 + w[1] * C1

        Cm2, Cm1 = C0, C1
        for k in range(2, self.K):
            kf = float(k)
            a = 2.0 * (kf + lam - 1.0) / kf
            b = (kf + 2.0 * lam - 2.0) / kf
            Ck = a * x * Cm1 - b * Cm2
            out = out + w[k] * Ck
            Cm2, Cm1 = Cm1, Ck

        return self.g * out

    def get_params(self):
        w = self._degree_weights()
        w = w / self.C_scaler
        w = self.g * w
        return w.detach().cpu().numpy()


class MCCF(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(MCCF, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.max_degree = config['max_degree']
        self.beta = config['beta']
        self.gain = config['gain']

        self.activate = LearnableGegenbauer(
            lam = (self.embedding_size - 2) / 2,
            max_degree = self.max_degree,
            g = self.gain
        )

        # self.activate = lambda x: self.gain * x
        # self.activate = lambda x: torch.log(torch.exp(self.gain * x) + torch.exp(self.gain * x ** 2))

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def norm_user_embedding(self, user_ids):
        user_embeds = self.user_embedding(user_ids)
        return F.normalize(user_embeds, dim=-1)

    def norm_item_embedding(self, item_ids):
        item_embeds = self.item_embedding(item_ids)
        return F.normalize(item_embeds, dim=-1)
    
    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)

    def calculate_loss(self, interaction):
        user_ids = interaction[self.USER_ID]
        item_ids = interaction[self.ITEM_ID]
        user_embeds = self.norm_user_embedding(user_ids)
        item_embeds = self.norm_item_embedding(item_ids)
        
        pos = (user_embeds * item_embeds).sum(dim=-1)
        pos = self.activate(pos)

        uni_items = torch.unique(item_ids)
        uni_embeds = self.norm_item_embedding(uni_items)
        uni = user_embeds @ uni_embeds.T
        uni = self.activate(uni)

        return - torch.log(torch.exp(pos) / torch.exp(uni).sum(dim=1)).mean() \
            - self.beta * self.activate.entropy()

            
    def predict(self, interaction):
        return

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_embeds = self.user_embedding(user)
        item_embeds = self.item_embedding.weight
        user_norms = torch.norm(user_embeds, dim=-1, keepdim=True)
        item_norms = torch.norm(item_embeds, dim=-1, keepdim=True)
        user_units = F.normalize(user_embeds, dim=-1)
        item_units = F.normalize(item_embeds, dim=-1)
        cos_scores = user_units @ item_units.T
        magnitudes = user_norms @ item_norms.T
        return self.activate(cos_scores) * magnitudes