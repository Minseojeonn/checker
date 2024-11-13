import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
from loguru import logger
import torch

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
            device
    ) -> None:
        self.dataset_name = dataset_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.shuffle = shuffle
        self.device = device
        
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
        breakpoint()
        raise NotImplementedError

    def data_split(self):
        """
        Split the dataset into train, validation, and test set
        """
        if self.shuffle:
            self.preprocessed_data = self.preprocessed_data.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        train_size = int(len(self.preprocessed_data) * self.train_ratio)
        val_size = int(len(self.preprocessed_data) * self.val_ratio)
        
        train_data = self.preprocessed_data[:train_size]
        val_data = self.preprocessed_data[train_size:train_size+val_size]
        test_data = self.preprocessed_data[train_size+val_size:]
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def data_reindexing(self):
        """
        Reindexing the dataset
        # user_id, item_id each starts from 0
        """
        user_id = list(set(self.preprocessed_data['user_id']))
        user_id2idx = {old: new for new, old in enumerate(user_id)}
        self.preprocessed_data['user_id'] = list(map(lambda x : user_id2idx[x], self.preprocessed_data['user_id']))

        item_id = list(set(self.preprocessed_data['item_id']))
        item_id = {old: new for new, old in enumerate(item_id)}
        self.preprocessed_data['item_id'] = list(map(lambda x : user_id2idx[x], self.preprocessed_data['item_id']))


    
    