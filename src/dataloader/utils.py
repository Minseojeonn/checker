import os
import numpy as np

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




