"""
Author: jinsong jsong010123@gmail.com
Date: 2024-09-23 13:56:29
LastEditTime: 2024-09-23 14:03:06
FilePath: /jinsong/code/work/glcc-hugegraph/diy/diy_hugegraph2dgl.py
Description: 

Copyright (c) 2024 by jinsong, All Rights Reserved. 
"""

from pyhugegraph.client import PyHugeClient
import torch
import dgl
import numpy as np
import pandas as pd

def hugegraph2dgl(client):
    g = client.gremlin()

    # 1. construct dgl graph
    vertex_map = {}  # original id -> id(int, from 0)
    edge_list = []
    edges = g.exec("g.E()")
    for edge in edges["data"]:
        # print(edge)
        src_orig = edge["outV"]
        dest_orig = edge["inV"]
        if src_orig not in vertex_map:
            vertex_map[src_orig] = len(vertex_map)
        src = vertex_map[src_orig]
        if dest_orig not in vertex_map:
            vertex_map[dest_orig] = len(vertex_map)
        dest = vertex_map[dest_orig]
        edge_list.append((src, dest))

    src_list, dest_list = zip(*edge_list)
    dgl_g = dgl.graph((torch.tensor(src_list), torch.tensor(dest_list)))

    # 2. generate features
    v_num = len(vertex_map)
    print(v_num)
    feat_size = 100
    features = np.random.random((v_num, feat_size)).astype(np.float32)
    features = torch.tensor(features)
    
    # 3. generate labels
    class_num = 7
    labels = np.random.randint(class_num, size=v_num)
    labels = torch.tensor(labels)

    return dgl_g, features, labels


def split_dataset(dgl_g, train_percent, val_percent):
    # split dataset 7 : 1.5 : 1.5
    v_num = dgl_g.num_nodes()
    # print(f"v_num = {v_num}")
    v_ids = np.arange(v_num)
    np.random.shuffle(v_ids)
    train_len = int(v_num * train_percent)
    val_len = int(v_num * val_percent)
    test_len = v_num - train_len - val_len
    train_mask = np.zeros(v_num, dtype=int)
    train_mask[v_ids[0:train_len]] = 1
    val_mask = np.zeros(v_num, dtype=int)
    val_mask[v_ids[train_len : train_len + val_len]] = 1
    test_mask = np.zeros(v_num, dtype=int)
    test_mask[v_ids[train_len + val_len : v_num]] = 1
    
    train_mask = torch.tensor(train_mask)
    val_mask = torch.tensor(val_mask)
    test_mask = torch.tensor(test_mask)

    return train_mask, val_mask, test_mask


if __name__ == "__main__":
    client = PyHugeClient(
        "127.0.0.1",
        "8080",
        user="admin",
        pwd="admin",
        graph="hugegraph_diy",
        # graphspace=None,
    )

    dgl_g, features, labels = hugegraph2dgl(client=client)

    train_percent = 0.7
    val_percent = 0.15
    train_mask, val_mask, test_mask = split_dataset(
        dgl_g=dgl_g, train_percent=train_percent, val_percent=val_percent
    )

    print(dgl_g)
    print(features.shape)
    print(labels.shape)
    num_ones = np.count_nonzero(train_mask == 1)
    print(num_ones / dgl_g.num_nodes())
