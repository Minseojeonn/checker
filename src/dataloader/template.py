import numpy as np
from sklearn.model_selection import train_test_split
from dataloader.utils import adj_dict_to_src_dest
from loguru import logger
import torch
from dataloader.augmentation import add_vnes

class template(object):
    """
    Wrapper class for datasets
    """
    def __init__(
            self,
            name,
            pos_augment,
            neg_augment,
            val_ratio,
            train_ratio,
            test_ratio,
            shuffle,
            seed,
            device
    ) -> None:
        self.name = name
        self.pos_augment = pos_augment
        self.neg_augment = neg_augment
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.shuffle = shuffle
        self.device = device
        
        if self.val_ratio + self.train_ratio + self.test_ratio != 1:
            raise Exception("sum of (train,val,test) ratio is not 1")

        self.processing()

    def data_preprocessing(self):
        raise NotImplementedError

    def read_dataset(self, file_dir) -> dict:
        #Depend on dataset format 
        #abstract method
        raise NotImplementedError

    def save(self) -> None:
        """
        save dataset to pkl
        """
        raise NotImplementedError

    def data_split(self):
        raise NotImplementedError
        
    def processing(self):
        self.data_preprocessing()
        raise NotImplementedError


    
    