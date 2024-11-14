#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score

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

def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def batch_uniform_random_sampling_optimized(user_ids, pos_items, user_interactions, num_items, num_samples=1):
    """
    여러 유저에 대해 uniform random sampling으로 negative samples를 빠르게 추출하는 최적화된 함수입니다.
    
    Parameters:
    - user_ids (list): negative sampling을 할 대상 유저 ID 리스트
    - pos_items (list): pos items
    - user_interactions (dict): 유저별 상호작용 아이템 dictionary, {user_id: [item1, item2, ...]}
    - num_items (int): 전체 아이템 개수
    - num_samples (int): 각 유저별 추출할 negative sample의 개수
    
    Returns:
    - batch_negative_samples (dict): {user_id: [negative sample 리스트]} 형태의 dictionary
    """
    new_neg_items = []
    new_pos_items = []
    new_user_ids = []
    all_items = set([i for i in range(num_items)])
    for user_id, pos_item_id in zip(user_ids, pos_items):
        # 유저가 상호작용한 아이템 집합을 미리 계산
        interacted_items = set(user_interactions.get(user_id, []))
        negative_candidates = list(all_items - interacted_items)
        
        # 유저별 샘플링 수행 (벡터화된 numpy 연산 사용)
        if len(negative_candidates) < num_samples:
            raise ValueError(f"유저 {user_id}에 대해 가능한 negative sampling 개수가 부족합니다.")
            
        # Negative samples를 무작위로 선택
        negative_samples = np.random.choice(negative_candidates, num_samples, replace=False)
        
        # 결과 저장
        for neg_item_id in negative_samples.tolist():
            new_user_ids.append(user_id)
            new_pos_items.append(pos_item_id)
            new_neg_items.append(neg_item_id)
    new_pos_items = torch.LongTensor(new_pos_items)
    new_neg_items = torch.LongTensor(new_neg_items)
    new_user_ids = torch.LongTensor(new_user_ids)
    return new_user_ids, new_pos_items, new_neg_items




# ====================Metrics==============================
# =========================================================


def RecallPrecision_ATk(test_data, recommended_indices, k):
    """
    Compute Recall and Precision at K.
    """
    test_users = test_data['user_id']
    test_items = test_data['item_id']
    recalls = []
    precisions = []

    for user, recommended in zip(test_users, recommended_indices):
        actual_items = test_items[test_data['user_id'] == user].tolist()  # 실제 아이템들
        recommended = recommended[:k].tolist()  # 상위 k개의 추천 아이템들

        # 교집합: 추천 아이템 중 실제로 사용자가 본 아이템
        hits = len(set(actual_items).intersection(set(recommended)))
        
        recall = hits / len(actual_items) if actual_items else 0
        precision = hits / k if k else 0
        
        recalls.append(recall)
        precisions.append(precision)

    # 평균 Recall, Precision
    return sum(recalls) / len(recalls), sum(precisions) / len(precisions)

def NDCGatK_r(test_data, recommended_indices, k):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at K.
    """
    test_users = test_data['user_id']
    test_items = test_data['item_id']
    ndcgs = []

    for user, recommended in zip(test_users, recommended_indices):
        actual_items = test_items[test_data['user_id'] == user].tolist()
        recommended = recommended[:k].tolist()
        
        dcg = 0.0
        idcg = 0.0

        # DCG 계산
        for rank, item in enumerate(recommended):
            if item in actual_items:
                dcg += 1 / torch.log2(torch.tensor(rank + 2.0))  # log2(rank+2)로 0회피

        # iDCG 계산 (이론적으로 가능한 최적의 DCG)
        for rank in range(min(len(actual_items), k)):
            idcg += 1 / torch.log2(torch.tensor(rank + 2.0))
        
        ndcg = (dcg / idcg).item() if idcg > 0 else 0
        ndcgs.append(ndcg)

    return sum(ndcgs) / len(ndcgs)