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
import warnings
from typing import Optional

import dgl
import torch
from pyhugegraph.api.gremlin import GremlinManager
from pyhugegraph.client import PyHugeClient


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
        info_vertex_label: str,
        vertex_label: str,
        edge_label: str,
        feat_key: str = "feat",
        label_key: str = "label",
    ):
        info_vertex = self._graph_germlin.exec(f"g.V().hasLabel('{info_vertex_label}')")["data"]
        vertices = self._graph_germlin.exec(f"g.V().hasLabel('{vertex_label}')")["data"]
        edges = self._graph_germlin.exec(f"g.E().hasLabel('{edge_label}')")["data"]

        graph_dgl = self._convert_graph_from_v_e(vertices, edges, feat_key, label_key)

        train_mask = info_vertex[0]["properties"]["train_mask"]
        val_mask = info_vertex[0]["properties"]["val_mask"]
        test_mask = info_vertex[0]["properties"]["test_mask"]
        graph_dgl.ndata["train_mask"] = torch.tensor(train_mask, dtype=torch.bool)
        graph_dgl.ndata["val_mask"] = torch.tensor(val_mask, dtype=torch.bool)
        graph_dgl.ndata["test_mask"] = torch.tensor(test_mask, dtype=torch.bool)

        graph_dgl_info = {
            "n_nodes": graph_dgl.number_of_nodes(),
            "n_edges": graph_dgl.number_of_edges(),
            "n_classes": graph_dgl.ndata["label"].unique().shape[0],
            "n_feat_dim": graph_dgl.ndata["feat"].size()[1],
        }

        return graph_dgl, graph_dgl_info

    def convert_graphs(
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
        print(graphs_info)
        return graphs, graphs_info

    @staticmethod
    def _convert_graph_from_v_e(vertices, edges, feat_key=None, label_key=None):
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

        return graph_dgl


if __name__ == "__main__":
    hg2d = HugeGraph2DGL()
    hg2d.convert_graphs(
        graph_vertex_label="MUTAG_graph_vertex",
        vertex_label="MUTAG_vertex",
        edge_label="MUTAG_edge",
    )
