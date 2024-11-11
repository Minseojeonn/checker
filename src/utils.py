#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
from loguru import logger
import scipy.sparse

def set_random_seed(seed, device):
    # for reproducibility (always not guaranteed in pytorch)
    # [1] https://pytorch.org/docs/stable/notes/randomness.html
    # [2] https://hoya012.github.io/blog/reproducible_pytorch/

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def log_param(param):
    for key, value in param.items():
        if type(value) is dict:
            for in_key, in_value in value.items():
                logger.info('{:20}:{:>50}'.format(in_key, '{}'.format(in_value)))
        else:
            logger.info('{:20}:{:>50}'.format(key, '{}'.format(value)))


def build_adj_matrix_uni_to_bi(train_data): #gowalla용으로 하드코딩 된것이라서, 나중에 다 바꿔줘야함
    train_data_tensor = torch.tensor(train_data).clone().t()
    indices = train_data_tensor[:2]
    indices[-1] = indices[-1]-29858
    values = torch.ones(indices.shape[-1], dtype=torch.long)
    bipartite_train_adj = torch.sparse_coo_tensor(indices, values, (29858, 40981)) #하드코딩 나중에 제거해야함.
    indices = bipartite_train_adj._indices().numpy()
    values = bipartite_train_adj._values().numpy()
    size = bipartite_train_adj.size()
    return scipy.sparse.coo_matrix((values, (indices[0], indices[1])), shape=size)


    
