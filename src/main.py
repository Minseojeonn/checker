#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import fire
import torch
from pathlib import Path
from utils import set_random_seed
from utils import log_param
from dataloader.utils import dataAdaper
from loguru import logger


def main(
        model='SDGNN',
        seed=1,
        epochs=100,
        learning_rate=0.001,
        device='cuda:0',
        dataset="gowalla",
        data_path="../data/gowalla",
        in_dim=20,
        out_dim=20,
        layer_num=3
    ):
    """
    Handle user arguments of ml-project-template

    :param model: name of model to be trained and tested
    :param seed: random_seed (if -1, a default seed is used)
    :param epochs: number of training epochs
    :param learning_rate: learning rate
    """
    
    # Step 0. Initialization
    logger.info("The main procedure has started with the following parameters:")
    device = device if torch.cuda.is_available() else 'cpu'
    
    set_random_seed(seed=seed, device=device)
    param = dict()
    param['model'] = model
    param['seed'] = seed
    param['device'] = device
    log_param(param)

    # Step 1. Load datasets
    dataset_list = ["gowalla"]
    if dataset.lower() not in dataset_list:
        raise Exception("not supported dataset.")
    data = dataAdaper(dataset.lower()) 


    ----------------------------
    # Step 2. Run (train and evaluate) the specified model
    logger.info("Training the model has begun with the following hyperparameters:")
    hyper_param = dict()
    hyper_param['epochs'] = epochs
    hyper_param['learning_rate'] = learning_rate
    hyper_param['in_dim'] = in_dim
    hyper_param['out_dim'] = out_dim
    hyper_param['layer_num'] = layer_num
    log_param(hyper_param)

    #eval
    eval_dict = {"sdgnn":SDGNNeval}
    evaluator = eval_dict[model.lower()](device=device)
    
    model_dict = {"sdgnn":run_sdgnn}
    if model.lower() not in model_dict:
        raise Exception("not supported model.")
    
    test_recall, test_hit = model_dict[model.lower()](
                        device=device,
                        dataset=dataset,
                        hyper_param=hyper_param,
                        evaluator=evaluator
                    )



    # Step 3. Report and save the final results
    logger.info("The model has been trained. The test recall is {:.4}. / hit is {:.4}".format(test_recall, test_hit))


if __name__ == "__main__":
    sys.exit(fire.Fire(main))