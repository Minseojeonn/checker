import os
import numpy as np


def dataAdaper(dataset):
    raise NotImplementedError

def get_node_degrees(edges):
    d = {}
    for edge in edges:
        if d.get(edge[0]) is None:
            d[edge[0]] = 0
        if d.get(edge[1]) is None:
            d[edge[1]] = 0
        d[edge[0]] += 1
        d[edge[1]] += 1
    return d

def get_in_out_degrees(edges):
    d_out = {}
    d_in = {}
    for edge in edges:
        if d_out.get(edge[0]) is None:
            d_out[edge[0]] = 0
        if d_out.get(edge[1]) is None:
            d_out[edge[1]] = 0
        if d_in.get(edge[0]) is None:
            d_in[edge[0]] = 0
        if d_in.get(edge[1]) is None:
            d_in[edge[1]] = 0

        d_out[edge[0]] += 1
        d_in[edge[1]] += 1
    return d_in, d_out

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory - " + directory)

def adj_dict_to_src_dest(adj_dict):
    src_dest = []
    max_user_idx = -1
    max_item_idx = -1
    for src in adj_dict:
        max_user_idx = max(max_user_idx, int(src))
        for dest in adj_dict[src]:
            src_dest.append([int(src),int(dest)])
            max_item_idx = max(int(dest), max_item_idx)
    return src_dest, max_user_idx, max_item_idx



