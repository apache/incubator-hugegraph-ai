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

# pylint: disable=too-many-branches

import warnings
from typing import Optional, List

import dgl
import torch
from pyhugegraph.api.gremlin import GremlinManager
from pyhugegraph.client import PyHugeClient

from hugegraph_ml.data.hugegraph_dataset import HugeGraphDataset


class HugeGraph2DGL:
    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: str = "8080",
        graph: str = "hugegraph",
        user: str = "",
        pwd: str = "",
        graphspace: Optional[str] = None,
    ):
        self._client: PyHugeClient = PyHugeClient(
            ip=ip, port=port, graph=graph, user=user, pwd=pwd, graphspace=graphspace
        )
        self._graph_germlin: GremlinManager = self._client.gremlin()

    def convert_graph(
        self,
        vertex_label: str,
        edge_label: str,
        feat_key: str = "feat",
        label_key: str = "label",
        mask_keys: Optional[List[str]] = None,
    ):
        if mask_keys is None:
            mask_keys = ["train_mask", "val_mask", "test_mask"]
        vertices = self._graph_germlin.exec(f"g.V().hasLabel('{vertex_label}')")["data"]
        edges = self._graph_germlin.exec(f"g.E().hasLabel('{edge_label}')")["data"]
        graph_dgl = self._convert_graph_from_v_e(vertices, edges, feat_key, label_key, mask_keys)

        return graph_dgl

    def convert_hetero_graph(
        self,
        vertex_labels: List[str],
        edge_labels: List[str],
        feat_key: str = "feat",
        label_key: str = "label",
        mask_keys: Optional[List[str]] = None,
    ):
        if mask_keys is None:
            mask_keys = ["train_mask", "val_mask", "test_mask"]
        vertex_label_id2idx = {}
        vertex_label_data = {}
        # for each vertex label
        for vertex_label in vertex_labels:
            vertices = self._graph_germlin.exec(f"g.V().hasLabel('{vertex_label}')")["data"]
            if len(vertices) == 0:
                warnings.warn(f"Graph has no vertices of vertex_label: {vertex_label}", Warning)
            else:
                vertex_ids = [v["id"] for v in vertices]
                id2idx = {vertex_id: idx for idx, vertex_id in enumerate(vertex_ids)}
                vertex_label_id2idx[vertex_label] = id2idx
                # extract vertex property(feat, label, mask)
                vertex_label_data[vertex_label] = {}
                if feat_key in vertices[0]["properties"]:
                    node_feats = torch.tensor([v["properties"][feat_key] for v in vertices], dtype=torch.float32)
                    vertex_label_data[vertex_label]["feat"] = node_feats
                if label_key in vertices[0]["properties"]:
                    node_labels = torch.tensor([v["properties"][label_key] for v in vertices], dtype=torch.long)
                    vertex_label_data[vertex_label]["label"] = node_labels
                if mask_keys:
                    for mk in mask_keys:
                        if mk in vertices[0]["properties"]:
                            mask = torch.tensor([v["properties"][mk] for v in vertices], dtype=torch.bool)
                            vertex_label_data[vertex_label][mk] = mask
        # build hetero graph from edges
        edge_data_dict = {}
        for edge_label in edge_labels:
            edges = self._graph_germlin.exec(f"g.E().hasLabel('{edge_label}')")["data"]
            if len(edges) == 0:
                warnings.warn(f"Graph has no edges of edge_label: {edge_label}", Warning)
            else:
                src_vertex_label = edges[0]["outVLabel"]
                src_idx = [vertex_label_id2idx[src_vertex_label][e["outV"]] for e in edges]
                dst_vertex_label = edges[0]["inVLabel"]
                dst_idx = [vertex_label_id2idx[dst_vertex_label][e["inV"]] for e in edges]
                edge_data_dict[(src_vertex_label, edge_label, dst_vertex_label)] = (src_idx, dst_idx)
        # add vertex properties data
        hetero_graph = dgl.heterograph(edge_data_dict)
        for vertex_label in vertex_labels:
            for prop in vertex_label_data[vertex_label]:
                hetero_graph.nodes[vertex_label].data[prop] = vertex_label_data[vertex_label][prop]

        return hetero_graph


    def convert_graph_dataset(
        self,
        graph_vertex_label: str,
        vertex_label: str,
        edge_label: str,
        feat_key: str = "feat",
        label_key: str = "label",
    ):
        # get graph vertices
        graph_vertices = self._graph_germlin.exec(f"g.V().hasLabel('{graph_vertex_label}')")["data"]
        graphs = []
        max_n_nodes = 0
        graph_labels = []
        for graph_vertex in graph_vertices:
            graph_id = graph_vertex["id"]
            label = graph_vertex["properties"][label_key]
            graph_labels.append(label)
            # get this graph's vertices and edges
            vertices = self._graph_germlin.exec(
                f"g.V().hasLabel('{vertex_label}').has('graph_id', {graph_id})")["data"]
            edges = self._graph_germlin.exec(
                f"g.E().hasLabel('{edge_label}').has('graph_id', {graph_id})")["data"]
            graph_dgl = self._convert_graph_from_v_e(vertices, edges, feat_key)
            graphs.append(graph_dgl)
            # record max num of node
            max_n_nodes = max(max_n_nodes, graph_dgl.number_of_nodes())
        # record dataset info
        graphs_info = {
            "n_graphs": len(graph_vertices),
            "max_n_nodes": max_n_nodes,
            "n_feat_dim": graphs[0].ndata["feat"].size()[1],
            "n_classes": len(set(graph_labels)),
        }
        dataset_dgl = HugeGraphDataset(graphs=graphs, labels=graph_labels, info=graphs_info)
        return dataset_dgl

    @staticmethod
    def _convert_graph_from_v_e(vertices, edges, feat_key=None, label_key=None, mask_keys=None):
        if len(vertices) == 0:
            warnings.warn("This graph has no vertices", Warning)
            return dgl.graph(())
        vertex_ids = [v["id"] for v in vertices]
        vertex_id_to_idx = {vertex_id: idx for idx, vertex_id in enumerate(vertex_ids)}
        src_idx = [vertex_id_to_idx[e["outV"]] for e in edges]
        dst_idx = [vertex_id_to_idx[e["inV"]] for e in edges]
        graph_dgl = dgl.graph((src_idx, dst_idx))

        if feat_key and feat_key in vertices[0]["properties"]:
            node_feats = [v["properties"][feat_key] for v in vertices]
            graph_dgl.ndata["feat"] = torch.tensor(node_feats, dtype=torch.float32)
        if label_key and label_key in vertices[0]["properties"]:
            node_labels = [v["properties"][label_key] for v in vertices]
            graph_dgl.ndata["label"] = torch.tensor(node_labels, dtype=torch.long)
        if mask_keys:
            for mk in mask_keys:
                if mk in vertices[0]["properties"]:
                    node_masks = [v["properties"][mk] for v in vertices]
                    mask = torch.tensor(node_masks, dtype=torch.bool)
                    graph_dgl.ndata[mk] = mask
        return graph_dgl


if __name__ == "__main__":
    hg2d = HugeGraph2DGL()
    hg2d.convert_graph(vertex_label="CORA_vertex", edge_label="CORA_edge")
    hg2d.convert_graph_dataset(
        graph_vertex_label="MUTAG_graph_vertex",
        vertex_label="MUTAG_vertex",
        edge_label="MUTAG_edge",
    )
    hg2d.convert_hetero_graph(
        vertex_labels=["ACM_paper_v", "ACM_author_v", "ACM_field_v"],
        edge_labels=["ACM_ap_e", "ACM_fp_e", "ACM_pa_e", "ACM_pf_e"]
    )
