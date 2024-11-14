import torch
from tqdm import tqdm
from utils import log_param, minibatch, batch_uniform_random_sampling_optimized, RecallPrecision_ATk, NDCGatK_r
from loguru import logger
from itertools import chain
from copy import deepcopy
from models.LightGCN.LightGCN import LightGCN
class LightgcnTrainer:
    def __init__(
            self
        ):
        pass

    def test(test_data, model, Ks, seen_items):
        """
        Test the performance of a trained model on test data
        """
        model.eval()
        with torch.no_grad():
            test_users = test_data['user_id']
            recalls = []
            precisions = []
            ndcgs = []
            rating_map = model.get_users_rating(test_users)
            
            for i, user in enumerate(test_users):
                if user in seen_items:
                    rating_map[i, seen_items[user]] = float('-inf')
            
            for k in Ks:
                recommended_indices = torch.topk(rating_map, k=k).indices
                recall, precision = RecallPrecision_ATk(test_data, recommended_indices, k)
                ndcg = NDCGatK_r(test_data, recommended_indices, k)
                recalls.append(recall)
                precisions.append(precision)
                ndcgs.append(ndcg)
                  
        return recall, precision, ndcg
    
    def train_with_hyper_param(self, dataset, hyper_param):
        #hyper parameters
        self.epochs = hyper_param['epochs']
        self.learning_rate = hyper_param['learning_rate']
        self.in_dim = hyper_param['in_dim']
        self.out_dim = hyper_param['out_dim']
        self.layer_num = hyper_param['layer_num']
        self.batch_size = hyper_param['batch_size']
        self.device = hyper_param['device']
        self.Ks = hyper_param['Ks']

        #model init
        self.model = LightGCN(
            config=hyper_param,
            dataset = dataset
        ).to(self.device)
        
        #score check
        best_score = 0
        best_model = None
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        #training phase
        self.model.train()
        train_data = dataset.get_data("train", self.device)
        val_data = dataset.get_data("val", self.device)
        
        val_dict = {
            "recall": [],
            "precision": [],
            "ndcg": []
            }
        test_dict = {
            "recall": [],
            "precision": [],
            "ndcg": []
        }
        
        for epoch in range(self.epochs):
            new_user_ids, new_pos_items, new_neg_items = batch_uniform_random_sampling_optimized(train_data["user_id"].cpu().numpy().tolist(), train_data["item_id"].cpu().numpy().tolist() ,dataset.get_all_pos_item(), dataset.num_items(), num_samples=1)
            self.model.train()
            for (batch_i,
                (batch_users,
                batch_pos,
                batch_neg)) in enumerate(minibatch(new_user_ids.to(self.device),
                                                new_pos_items.to(self.device),
                                                new_neg_items.to(self.device),
                                                batch_size=self.batch_size)):
                optimizer.zero_grad()
                loss = self.model.bpr_loss(batch_users, batch_pos, batch_neg)
                loss.backward()
                optimizer.step()
            
            #validation phase
            self.model.eval()
            recall, precision, ndcg = self.test(val_data, self.model, self.Ks, dataset.get_all_pos_item())
            val_dict["recall"].append(recall)
            val_dict["precision"].append(precision)
            val_dict["ndcg"].append(ndcg)
            
        
            test_recall, test_precision, test_ndcg = self.test(dataset.get_data("test", self.device), self.model, self.Ks, dataset.get_all_pos_item())
            test_dict["recall"].append(test_recall)
            test_dict["precision"].append(test_precision)
            test_dict["ndcg"].append(test_ndcg)
            
            #save best model
            if best_score < recall:
                best_score = recall
                best_model = deepcopy(self.model.state_dict())
        
        return best_model
    

    
def run_lightgcn(dataset, hyper_param):
    #trainer
    trainer = LightgcnTrainer()
    model = trainer.train_with_hyper_param(
        dataset=dataset,
        hyper_param=hyper_param
        )
    
    return model

