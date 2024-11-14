import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
from loguru import logger
import torch
import json
from dataloader.utils import convert_list_in_dict_to_tensor

class template(object):
    """
    Wrapper class for datasets
    """
    def __init__(
            self,
            dataset_name,
            val_ratio,
            train_ratio,
            test_ratio,
            shuffle,
            seed,
            device,
            model
    ) -> None:
        self.dataset_name = dataset_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.shuffle = shuffle
        self.device = device
        self.model = model

        if self.val_ratio + self.train_ratio + self.test_ratio != 1:
            raise Exception("sum of (train,val,test) ratio is not 1")
        self.load_dataset()
        self.processing()

    def data_preprocessing(self):
        """
        data preprocessing
        abastrct method
        
        Returns: dict of preprocessed data 
            keys: user_id, item_id, rating, timestamp
        """
        raise NotImplementedError

    def load_dataset(self):
        """
        Load dataset from file
        """
        preprocessed_data_name = f"./processed_data/{self.dataset_name}.pkl"
        if os.path.isfile(preprocessed_data_name): #file exists
            logger.info("Preprocessed data exists")
            self.preprocessed_data = pickle.load(open(preprocessed_data_name, "rb"))

        else: #no file
            if os.path.isdir("./processed_data") == False:
                os.makedirs("./processed_data")
            logger.info("Preprocessed data not exists")
            self.preprocessed_data = self.data_preprocessing()
            self.data_reindexing()
            pickle.dump(self.preprocessed_data, open(preprocessed_data_name, "wb"))

    def processing(self): #processing data for oveall dataset 
        
        with open("./dataloader/model_setting.json", "r") as f:
            config_json = json.load(f)
            config_json = config_json[self.model.lower()]
        using_sign = config_json['using_sign'] #boolean
        self.data_reindexing()
        self.data_split()
        self.create_sparse_graph(using_sign)
        self.nomalizing_graph()

    def data_split(self):
        """
        Split the dataset into train, validation, and test set
        """
        if self.shuffle:
            import random
            temp_idx = [i for i in range(len(self.preprocessed_data["user_id"]))]
            random.shuffle(temp_idx)
            self.preprocessed_data["user_id"] = [self.preprocessed_data["user_id"][idx] for idx in temp_idx]
            self.preprocessed_data["item_id"] = [self.preprocessed_data["item_id"][idx] for idx in temp_idx]
            self.preprocessed_data["rating"] = [self.preprocessed_data["rating"][idx] for idx in temp_idx]
            self.preprocessed_data["timestamp"] = [self.preprocessed_data["timestamp"][idx] for idx in temp_idx]
            self.preprocessed_data["sign"] = [self.preprocessed_data["sign"][idx] for idx in temp_idx]
        
        train_size = int(len(self.preprocessed_data["user_id"]) * self.train_ratio)
        val_size = int(len(self.preprocessed_data["user_id"]) * self.val_ratio)
        
        train_data = {
            "user_id": self.preprocessed_data["user_id"][:train_size],
            "item_id": self.preprocessed_data["item_id"][:train_size],
            "rating": self.preprocessed_data["rating"][:train_size],
            "timestamp": self.preprocessed_data["timestamp"][:train_size],
            "sign": self.preprocessed_data["sign"][:train_size]   
            }
        val_data = {
            "user_id": self.preprocessed_data["user_id"][train_size:train_size+val_size],
            "item_id": self.preprocessed_data["item_id"][train_size:train_size+val_size],
            "rating": self.preprocessed_data["rating"][train_size:train_size+val_size],
            "timestamp": self.preprocessed_data["timestamp"][train_size:train_size+val_size],
            "sign": self.preprocessed_data["sign"][train_size:train_size+val_size]
            }
        test_data = {
            "user_id": self.preprocessed_data["user_id"][train_size+val_size:],
            "item_id": self.preprocessed_data["item_id"][train_size+val_size:],
            "rating": self.preprocessed_data["rating"][train_size+val_size:],
            "timestamp": self.preprocessed_data["timestamp"][train_size+val_size:],
            "sign": self.preprocessed_data["sign"][train_size+val_size:]
            }
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def get_data(self, type, device):
        """
        Get the data for training, validation, and test
        """
        if type == "train":
            return convert_list_in_dict_to_tensor(self.train_data, device)
        elif type == "val":
            return convert_list_in_dict_to_tensor(self.val_data, device)
        elif type == "test":
            return convert_list_in_dict_to_tensor(self.test_data, device)
        else:
            raise Exception("Invalid type")
    
    def data_reindexing(self):
        """
        Reindexing the dataset
        # user_id, item_id each starts from 0
        """
        user_id = set(self.preprocessed_data['user_id'])
        user_id = list(user_id)
        user_id2idx = {old: new for new, old in enumerate(user_id)}
        self.num_users = len(user_id2idx)
        self.preprocessed_data['user_id'] = list(map(lambda x : user_id2idx[x], self.preprocessed_data['user_id']))

        item_id = set(self.preprocessed_data['item_id'])
        item_id = list(item_id)
        item_id2idx = {old: new for new, old in enumerate(item_id)}
        self.num_items = len(item_id2idx)
        self.preprocessed_data['item_id'] = list(map(lambda x : item_id2idx[x], self.preprocessed_data['item_id']))

    def get_all_pos_item(self):
        """
        Get all positive items for a user
        """
        return self.user_pos_dict
    
    def create_sparse_graph(self, sign):
        """
        create sparse graph
        args:
            sign(boolean):
                True:
                    | 0        R |
                    | R.T      0 |
                False:
                    |   0    |R| |
                    | |R.T|   0  |
        """
        user_dim = torch.LongTensor(self.train_data['user_id'])
        item_dim = torch.LongTensor(self.train_data['item_id'])
        user_pos_dict = {i : [] for i in range(self.num_users)}
        for u, i in zip(self.train_data["user_id"], self.train_data["item_id"]):
            user_pos_dict[u].append(i)
        self.user_pos_dict = user_pos_dict    
        
        first_sub = torch.stack([user_dim, item_dim+self.num_users])
        second_sub = torch.stack([item_dim+self.num_users, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        if sign:
            data = torch.cat([self.train_data['sign'], self.train_data['sign']])
        else:
            data = torch.ones(index.size(-1)).int()
       
        assert (index.shape[-1] == len(data))
        
        self.Graph = torch.sparse.FloatTensor(index, data, torch.Size([self.num_users+self.num_items, self.num_users+self.num_items]))
        
    
    def nomalizing_graph(self):
        """
            Calculate the degree of each node in the graph.
            Args:
                graph (torch.sparse): torch.sparse matrix 
                |  0   R |
                | R.T  0 |
            Returns:
                dict: degree of each node in the graph
        """
        
        dense = self.Graph.to_dense()
        D = torch.sum(abs(dense), dim=1).float()
        D[D == 0.] = 1. #avoid division by zero
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense/D_sqrt.t() 
        index = dense.nonzero()
        data = dense[torch.logical_or(dense >= 1e-9, dense <= -1e-9)]
        assert len(index) == len(data)
        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.num_items+self.num_users, self.num_items+self.num_users]))
        self.Graph = Graph.coalesce()
    
    def getSparseGraph(self):
        return self.Graph