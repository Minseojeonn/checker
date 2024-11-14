from dataloader.template import template
import pandas as pd


def data_adapter(dataset_name):
    """
    Adapter function for dataset
    """
    if dataset_name == "ml-1m":
        return movielens_1m

class movielens_1m(template):
    """
    Wrapper class for datasets
    """
    def __init__(
            self,
            param
    ) -> None:
        super(movielens_1m, self).__init__(
            dataset_name=param["dataset"],
            val_ratio=param["split_ratio"][1],
            train_ratio=param["split_ratio"][0],
            test_ratio=param["split_ratio"][2],
            shuffle=param["shuffle"],
            seed=param["seed"],
            device=param["device"],
            model=param["model"]
        )
        
    def data_preprocessing(self):
        """
        data preprocessing
        abastrct method
        
        Returns: dict of preprocessed data 
            keys: user_id, item_id, rating, timestamp
        """
        data = pd.read_csv(f"../data/ml-1m.txt", sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'])
        data_dict = {
                        'user_id': data['user_id'].tolist(),
                        'item_id': data['item_id'].tolist(),
                        'rating': data['rating'].tolist(),
                        'timestamp': data['timestamp'].tolist()
                    }
        sign = []
        for i in data_dict['rating']:
            if i > 3:
                sign.append(1)
            else:
                sign.append(0)
        data_dict['sign'] = sign
        
        return data_dict
        

        
