
"""
Author: jinsong jsong010123@gmail.com
Date: 2024-09-23 13:38:09
LastEditTime: 2024-09-23 13:42:34
FilePath: /jinsong/code/work/glcc-hugegraph/diy_hugegraph_init.py
Description: 

Copyright (c) 2024 by jinsong, All Rights Reserved. 
"""

from pyhugegraph.client import PyHugeClient
import torch
import dgl
import numpy as np
import pandas as pd


# the vertex only have id (no feature and label)
def schema_init(client):
    """schema"""
    schema = client.schema()

    schema.propertyKey("id").asText().ifNotExist().create()

    schema.vertexLabel("Paper").properties("id").usePrimaryKeyId().primaryKeys(
        "id"
    ).ifNotExist().create()

    schema.edgeLabel("Cites").sourceLabel("Paper").targetLabel(
        "Paper"
    ).ifNotExist().create()
    return schema


# only need client and edge_list
def data_init(client, edge_path):
    g = client.graph()
    f = open(edge_path, "r")
    vertices = []
    i = 0
    for line in f:
        src_orig, dest_orig = line.strip("\n").split()
        if src_orig not in vertices:
            g.addVertex("Paper", {"id": f"{src_orig}"})
            vertices.append(src_orig)
        if dest_orig not in vertices:
            g.addVertex("Paper", {"id": f"{dest_orig}"})
            vertices.append(dest_orig)
        if i % 500 == 0:
            print(f"{i} lines are processed")
        g.addEdge("Cites", f"1:{src_orig}", f"1:{dest_orig}", {})
        i += 1

    return g


if __name__ == "__main__":
    client = PyHugeClient(
        "127.0.0.1",
        "8080",
        user="admin",
        pwd="admin",
        graph="hugegraph_diy",
        # graphspace=None,
    )

    """schema"""
    schema = schema_init(client)

    print(schema.getVertexLabels())
    print(schema.getEdgeLabels())
    print(schema.getRelations())
    print()

    edge_path = "hugegraph-ml/dataset/cora/cora.cites"
    g = data_init(client, edge_path)
    print(g)
