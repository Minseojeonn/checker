import os
import numpy as np
import torch

def create_folder(directory):
    """
        Create a folder if it does not exist.
        Args:
            directory (str): path to the folder
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory - " + directory)



def convert_list_in_dict_to_tensor(list_in_dict, device):
    for li in list_in_dict:
        list_in_dict[li] = torch.tensor(list_in_dict[li], device=device)    
        
    