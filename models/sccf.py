# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class SCCF(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(SCCF, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.temp = config['temperature']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
    
    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        t = self.temp
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        u_idx, u_inv_idx, u_counts = torch.unique(user, return_counts=True, return_inverse=True)
        i_idx, i_inv_idx, i_counts = torch.unique(item, return_counts=True, return_inverse=True)
        u_counts, i_counts = u_counts.reshape(-1, 1).float(), i_counts.reshape(-1, 1).float()
        user_e, item_e = self.forward(user, item)
        ip = (user_e * item_e).sum(dim=1)
        up_score = (ip / t).exp() + (ip ** 2 / t).exp()
        up = up_score.log().mean()
        user_e, item_e = self.forward(u_idx, i_idx)
        sim_mat = user_e @ item_e.T
        score = (sim_mat / t).exp() + (sim_mat ** 2 / t).exp()
        down = (score * (u_counts @ i_counts.T)).mean().log()
        loss = -up + down
        return loss
            
    def predict(self, interaction):
        # user = interaction[self.USER_ID]
        # item = interaction[self.ITEM_ID]
        # user_e = self.user_embedding(user)
        # item_e = self.item_embedding(item)
        # return torch.mul(user_e, item_e).sum(dim=1)
        return

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)