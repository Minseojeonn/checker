#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
import multiprocessing
import mlflow

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

import numpy as np
import torch
from multiprocessing import Pool

def generate_negative_samples_for_user(user_id, user_interactions, all_items, num_samples):
    """
    단일 유저에 대해 negative samples를 생성하는 함수
    """
    interacted_items = set(user_interactions.get(user_id, []))
    negative_candidates = list(all_items - interacted_items)
    
    # Negative candidates가 num_samples보다 적을 경우 예외 처리
    if len(negative_candidates) < num_samples:
        raise ValueError(f"유저 {user_id}에 대해 가능한 negative sampling 개수가 부족합니다. "
                         f"가능한 아이템 수: {len(negative_candidates)}")
    
    # Negative samples를 무작위로 선택
    negative_samples = np.random.choice(negative_candidates, num_samples, replace=False)
    
    return user_id, negative_samples.tolist()

def batch_uniform_random_sampling_optimized(user_ids, user_interactions, num_items, num_samples=1):
    """
    여러 유저에 대해 uniform random sampling으로 negative samples를 빠르게 추출하는 최적화된 함수입니다.
    
    Parameters:
    - user_ids (list): negative sampling을 할 대상 유저 ID 리스트
    - user_interactions (dict): 유저별 상호작용 아이템 dictionary, {user_id: [item1, item2, ...]}
    - all_items (set): 전체 아이템 ID의 집합
    - num_samples (int): 각 유저별 추출할 negative sample의 개수
    
    Returns:
    - batch_negative_samples (dict): {user_id: [negative sample 리스트]} 형태의 dictionary
    """
    batch_negative_samples = {}
    all_items = set(range(num_items))

    # 멀티프로세싱을 사용하여 병렬 처리
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # 각 유저별로 negative 샘플링을 병렬로 수행
        try:
            results = pool.starmap(generate_negative_samples_for_user, 
                                  [(user_id, user_interactions, all_items, num_samples) for user_id in user_ids])
        except ValueError as e:
            print(f"에러 발생: {e}")
            return None, None, None
        
        # 결과를 dict 형태로 변환
        for user_id, negative_samples in results:
            batch_negative_samples[user_id] = negative_samples
    
    # 샘플들을 새로운 리스트로 변환
    new_neg_items = []
    new_pos_items = []
    new_user_ids = []
    for user_id in user_ids:
        pos_items = user_interactions.get(user_id, [])
        
        for pos_item in pos_items:
            new_user_ids.extend([user_id] * num_samples)
            new_pos_items.extend([pos_item] * num_samples)
            new_neg_items.extend(batch_negative_samples[user_id])
    
    length = len(new_user_ids)
    # Torch tensor로 변환
    new_pos_items = torch.LongTensor(new_pos_items)
    new_neg_items = torch.LongTensor(new_neg_items)
    new_user_ids = torch.LongTensor(new_user_ids)
    
    return new_user_ids, new_pos_items, new_neg_items, length




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


#------------------------------------------------mlflow-logger-------------------------------------------------

def logging_with_mlflow_metric(test, valid):
    # metric selected by valid score
    best_recall, best_precision, best_ndcg = -float('inf'), -float('inf'), -float('inf')
    best_recall_epoch, best_precision_epoch, best_ndcg_epoch = -1, -1, -1
    
    for idx in range(len(test["recall"])):
        if valid["recall"][idx] > best_recall:
            best_recall = valid["recall"][idx]
            best_recall_epoch = idx
        if valid["precision"][idx] > best_precision:
            best_precision = valid["precision"][idx]
            best_precision_epoch = idx
        if valid["ndcg"][idx] > best_ndcg:
            best_ndcg = valid["ndcg"][idx]
            best_ndcg_epoch = idx
        metrics_dict = {
            "test_recall": test["recall"][idx],
            "test_precision": test["precision"][idx],
            "test_ndcg": test["ndcg"][idx],
            "valid_recall": valid["recall"][idx],
            "valid_precision": valid["precision"][idx],
            "valid_ndcg": valid["ndcg"][idx]
        }
        mlflow.log_metrics(metrics_dict, synchronous=False, step=idx)

    best_metrics_dict = {
        "best_recall_test" : test["recall"][best_recall_epoch],
        "best_precision_test" : test["precision"][best_precision_epoch],
        "best_ndcg_test" : test["ndcg"][best_ndcg_epoch],
        "best_recall_valid" : valid["recall"][best_recall_epoch],
        "best_precision_valid" : valid["precision"][best_precision_epoch],
        "best_ndcg_valid" : valid["ndcg"][best_ndcg_epoch]
        }
    mlflow.log_metrics(best_metrics_dict, synchronous=True)
