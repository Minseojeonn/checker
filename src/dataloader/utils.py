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


def nomalizing_graph(graph, num_users, num_items):
    """
        Calculate the degree of each node in the graph.
        Args:
            graph (torch.sparse): torch.sparse matrix 
            |  0   R |
            | R.T  0 |
        Returns:
            dict: degree of each node in the graph
    """
    
    dense = graph.to_dense()
    D = torch.sum(dense, dim=1).float()
    D[D == 0.] = 1. #avoid division by zero
    D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
    dense = dense / D_sqrt
    dense = dense/D_sqrt.t() 
    index = dense.nonzero()
    data = dense[dense >= 1e-9 or dense <= -1e-9]
    assert len(index) == len(data)
    Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size[[num_items+num_users, num_items+num_users]])
    Graph = Graph.coalesce()
    return Graph
