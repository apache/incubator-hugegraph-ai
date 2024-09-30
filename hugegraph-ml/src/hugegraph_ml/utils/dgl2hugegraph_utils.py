# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
from typing import Optional

import dgl
import numpy as np
import scipy
import torch
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, LegacyTUDataset, GINDataset, \
    get_download_dir
from dgl.data.utils import _get_dgl_url, download
from pyhugegraph.api.graph import GraphManager
from pyhugegraph.api.schema import SchemaManager
from pyhugegraph.client import PyHugeClient


def clear_all_data(
    ip: str = "127.0.0.1",
    port: str = "8080",
    graph: str = "hugegraph",
    user: str = "",
    pwd: str = "",
    graphspace: Optional[str] = None,
):
    client: PyHugeClient = PyHugeClient(ip=ip, port=port, graph=graph, user=user, pwd=pwd, graphspace=graphspace)
    client.graphs().clear_graph_all_data()


def import_graph_from_dgl(
    dataset_name,
    ip: str = "127.0.0.1",
    port: str = "8080",
    graph: str = "hugegraph",
    user: str = "",
    pwd: str = "",
    graphspace: Optional[str] = None,
):
    dataset_name = dataset_name.upper()
    if dataset_name == "CORA":
        dataset_dgl = CoraGraphDataset()
    elif dataset_name == "CITESEER":
        dataset_dgl = CiteseerGraphDataset()
    elif dataset_name == "PUBMED":
        dataset_dgl = PubmedGraphDataset()
    else:
        raise ValueError("dataset not supported")
    graph_dgl = dataset_dgl[0]

    client: PyHugeClient = PyHugeClient(ip=ip, port=port, graph=graph, user=user, pwd=pwd, graphspace=graphspace)
    client_schema: SchemaManager = client.schema()
    client_graph: GraphManager = client.graph()

    client_schema.propertyKey("feat").asDouble().valueList().ifNotExist().create()
    client_schema.propertyKey("label").asLong().ifNotExist().create()
    client_schema.propertyKey("train_mask").asInt().ifNotExist().create()
    client_schema.propertyKey("val_mask").asInt().ifNotExist().create()
    client_schema.propertyKey("test_mask").asInt().ifNotExist().create()

    vertex_label = f"{dataset_name}_vertex"
    all_props = ["feat", "label", "train_mask", "val_mask", "test_mask"]
    props = [p for p in all_props if p in graph_dgl.ndata]
    props_value = {}
    for p in props:
        props_value[p] = graph_dgl.ndata[p].tolist()
    client_schema.vertexLabel(vertex_label).useAutomaticId().properties(*props).ifNotExist().create()
    idx_to_vertex_id = {}
    for idx in range(graph_dgl.number_of_nodes()):
        properties = {}
        for p in props:
            if isinstance(props_value[p][idx], bool):
                properties[p] = int(props_value[p][idx])
            else:
                properties[p] = props_value[p][idx]
        vertex = client_graph.addVertex(label=vertex_label, properties=properties)
        idx_to_vertex_id[idx] = vertex.id

    edge_label = f"{dataset_name}_edge"
    client_schema.edgeLabel(edge_label).sourceLabel(vertex_label).targetLabel(vertex_label).ifNotExist().create()
    edges_src, edges_dst = graph_dgl.edges()
    for src, dst in zip(edges_src.numpy(), edges_dst.numpy()):
        client_graph.addEdge(
            edge_label=edge_label,
            out_id=idx_to_vertex_id[src],
            in_id=idx_to_vertex_id[dst],
            properties={}
        )


def import_graphs_from_dgl(
    dataset_name,
    ip: str = "127.0.0.1",
    port: str = "8080",
    graph: str = "hugegraph",
    user: str = "",
    pwd: str = "",
    graphspace: Optional[str] = None,
):
    dataset_name = dataset_name.upper()
    # load dgl bultin dataset
    if dataset_name in ["ENZYMES", "DD"]:
        dataset_dgl = LegacyTUDataset(name=dataset_name)
    elif dataset_name in ["MUTAG", "COLLAB", "NCI1", "PROTEINS", "PTC"]:
        dataset_dgl = GINDataset(name=dataset_name, self_loop=True)
    else:
        raise ValueError("dataset not supported")
    # hugegraph client
    client: PyHugeClient = PyHugeClient(ip=ip, port=port, graph=graph, user=user, pwd=pwd, graphspace=graphspace)
    client_schema: SchemaManager = client.schema()
    client_graph: GraphManager = client.graph()
    # define vertexLabel/edgeLabel
    graph_vertex_label = f"{dataset_name}_graph_vertex"
    vertex_label = f"{dataset_name}_vertex"
    edge_label = f"{dataset_name}_edge"
    # create schema
    client_schema.propertyKey("label").asLong().ifNotExist().create()
    client_schema.propertyKey("feat").asDouble().valueList().ifNotExist().create()
    client_schema.propertyKey("graph_id").asLong().ifNotExist().create()
    client_schema.vertexLabel(graph_vertex_label).useAutomaticId().properties("label").ifNotExist().create()
    client_schema.vertexLabel(vertex_label).useAutomaticId().properties("feat", "graph_id").ifNotExist().create()
    client_schema.edgeLabel(edge_label).sourceLabel(vertex_label).targetLabel(vertex_label).properties(
        "graph_id").ifNotExist().create()
    client_schema.indexLabel("vertex_by_graph_id").onV(vertex_label).by("graph_id").secondary().ifNotExist().create()
    client_schema.indexLabel("edge_by_graph_id").onE(edge_label).by("graph_id").secondary().ifNotExist().create()
    # import to hugegraph
    for (graph_dgl, label) in dataset_dgl:
        graph_vertex = client_graph.addVertex(label=graph_vertex_label, properties={"label": int(label)})
        if "feat" in graph_dgl.ndata:
            node_feats = graph_dgl.ndata["feat"]
        elif "attr" in graph_dgl.ndata:
            node_feats = graph_dgl.ndata["attr"]
        else:
            raise ValueError("Node feature is empty")
        idx_to_vertex_id = {}
        for idx in range(graph_dgl.number_of_nodes()):
            feat = node_feats[idx].tolist()
            vertex = client_graph.addVertex(
                label=vertex_label,
                properties={"feat": feat, "graph_id": graph_vertex.id}
            )
            idx_to_vertex_id[idx] = vertex.id
        srcs, dsts = graph_dgl.edges()
        for src, dst in zip(srcs.numpy(), dsts.numpy()):
            client_graph.addEdge(
                edge_label=edge_label,
                out_id=idx_to_vertex_id[src],
                in_id=idx_to_vertex_id[dst],
                properties={"graph_id": graph_vertex.id}
            )


def import_hetero_graph_from_dgl(
    dataset_name,
    ip: str = "127.0.0.1",
    port: str = "8080",
    graph: str = "hugegraph",
    user: str = "",
    pwd: str = "",
    graphspace: Optional[str] = None,
):
    dataset_name = dataset_name.upper()
    if dataset_name == "ACM":
        hetero_graph = load_acm_raw()
    else:
        raise ValueError("dataset not supported")
    client: PyHugeClient = PyHugeClient(ip=ip, port=port, graph=graph, user=user, pwd=pwd, graphspace=graphspace)
    client_schema: SchemaManager = client.schema()
    client_graph: GraphManager = client.graph()

    client_schema.propertyKey("feat").asDouble().valueList().ifNotExist().create()
    client_schema.propertyKey("label").asLong().ifNotExist().create()
    client_schema.propertyKey("train_mask").asInt().ifNotExist().create()
    client_schema.propertyKey("val_mask").asInt().ifNotExist().create()
    client_schema.propertyKey("test_mask").asInt().ifNotExist().create()

    ntype_to_vertex_label = {}
    ntype_idx_to_vertex_id = {}
    for ntype in hetero_graph.ntypes:
        # create vertex schema
        vertex_label = f"{dataset_name}_{ntype}_v"
        ntype_to_vertex_label[ntype] = vertex_label
        all_props = ["feat", "label", "train_mask", "val_mask", "test_mask"]
        # check properties
        props = [p for p in all_props if p in hetero_graph.nodes[ntype].data]
        client_schema.vertexLabel(vertex_label).useAutomaticId().properties(*props).ifNotExist().create()
        # add vertices
        props_value = {}
        for p in props:
            props_value[p] = hetero_graph.nodes[ntype].data[p].tolist()
        # for each node
        idx_to_vertex_id = {}
        for idx in range(hetero_graph.number_of_nodes(ntype=ntype)):
            properties = {}
            # for each property
            for p in props:
                if isinstance(props_value[p][idx], bool):
                    properties[p] = int(props_value[p][idx])
                else:
                    properties[p] = props_value[p][idx]
            vertex = client_graph.addVertex(
                label=vertex_label,
                properties=properties
            )
            idx_to_vertex_id[idx] = vertex.id
        ntype_idx_to_vertex_id[ntype] = idx_to_vertex_id

    for canonical_etype in hetero_graph.canonical_etypes:
        # create edge schema
        src_type, etype, dst_type = canonical_etype
        edge_label = f"{dataset_name}_{etype}_e"
        client_schema.edgeLabel(edge_label).sourceLabel(ntype_to_vertex_label[src_type]).targetLabel(
            ntype_to_vertex_label[dst_type]
        ).ifNotExist().create()
        # add edges
        srcs, dsts = hetero_graph.edges(etype=canonical_etype)
        for src, dst in zip(srcs.numpy(), dsts.numpy()):
            client_graph.addEdge(
                edge_label=edge_label,
                out_id=ntype_idx_to_vertex_id[src_type][src],
                in_id=ntype_idx_to_vertex_id[dst_type][dst],
                properties={}
            )


def load_acm_raw():
    # reference: https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/utils.py
    url = "dataset/ACM.mat"
    data_path = get_download_dir() + "/ACM.mat"
    if not os.path.exists(data_path):
        print(f"File {data_path} not found, downloading...")
        download(_get_dgl_url(url), path=data_path)

    data = scipy.io.loadmat(data_path)
    p_vs_l = data["PvsL"]  # paper-field?
    p_vs_a = data["PvsA"]  # paper-author
    p_vs_t = data["PvsT"]  # paper-term, bag of words
    p_vs_c = data["PvsC"]  # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_selected = p_vs_c[:, conf_ids].tocoo().row
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    hgraph = dgl.heterograph(
        {
            ("paper", "pa", "author"): p_vs_a.nonzero(),
            ("author", "ap", "paper"): p_vs_a.transpose().nonzero(),
            ("paper", "pf", "field"): p_vs_l.nonzero(),
            ("field", "fp", "paper"): p_vs_l.transpose().nonzero(),
        }
    )

    features = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = pc_c == conf_id
        float_mask[pc_c_mask] = np.random.permutation(
            np.linspace(0, 1, pc_c_mask.sum())
        )
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hgraph.num_nodes("paper")
    train_mask = _get_mask(num_nodes, train_idx)
    val_mask = _get_mask(num_nodes, val_idx)
    test_mask = _get_mask(num_nodes, test_idx)

    hgraph.nodes["paper"].data["feat"] = features
    hgraph.nodes["paper"].data["label"] = labels
    hgraph.nodes["paper"].data["train_mask"] = train_mask
    hgraph.nodes["paper"].data["val_mask"] = val_mask
    hgraph.nodes["paper"].data["test_mask"] = test_mask

    return hgraph


def _get_mask(size, indices):
    mask = torch.zeros(size)
    mask[indices] = 1
    return mask.bool()


if __name__ == "__main__":
    clear_all_data()
    import_graph_from_dgl("CORA")
    import_graphs_from_dgl("MUTAG")
    import_hetero_graph_from_dgl("ACM")
