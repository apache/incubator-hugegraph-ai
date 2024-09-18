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

import dgl
import torch
from pyhugegraph.api.gremlin import GremlinManager
from pyhugegraph.client import PyHugeClient


class HugeGraph2DGL:
    def __init__(
            self,
            ip='127.0.0.1',
            port="8080",
            graph='hugegraph',
            user='admin',
            pwd='xxx'
    ):
        self._client: PyHugeClient = PyHugeClient(
            ip=ip,
            port=port,
            graph=graph,
            user=user,
            pwd=pwd
        )
        self._graph_germlin: GremlinManager = self._client.gremlin()

    def convert_graph(
            self,
            vertex_label,
            info_vertex_label,
            edge_label,
            feat_key="feat",
            label_key="label",
    ):
        vertices = self._graph_germlin.exec(f"g.V().hasLabel('{vertex_label}')")["data"]
        info_vertex = self._graph_germlin.exec(f"g.V().hasLabel('{info_vertex_label}')")["data"]
        edges = self._graph_germlin.exec(f"g.E().hasLabel('{edge_label}')")["data"]
        print(len(vertices), len(edges))

        node_ids = [v["id"] for v in vertices]
        node_feats = [v["properties"][feat_key] for v in vertices]
        node_labels = [v["properties"][label_key] for v in vertices]
        train_mask = info_vertex[0]["properties"]["train_mask"]
        val_mask = info_vertex[0]["properties"]["val_mask"]
        test_mask = info_vertex[0]["properties"]["test_mask"]

        src_ids = [e["outV"] for e in edges]
        dst_ids = [e["inV"] for e in edges]

        node_id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
        mapped_src_ids = [node_id_map[s] for s in src_ids]
        mapped_dst_ids = [node_id_map[d] for d in dst_ids]

        graph = dgl.graph((mapped_src_ids, mapped_dst_ids))
        graph.ndata["feat"] = torch.tensor(node_feats, dtype=torch.float32)
        graph.ndata["label"] = torch.tensor(node_labels, dtype=torch.long)
        graph.ndata["train_mask"] = torch.tensor(train_mask, dtype=torch.bool)
        graph.ndata["val_mask"] = torch.tensor(val_mask, dtype=torch.bool)
        graph.ndata["test_mask"] = torch.tensor(test_mask, dtype=torch.bool)

        graph_info = {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "n_classes": graph.ndata["label"].unique().shape[0],
            "n_feat_dim": graph.ndata["feat"].size()[1],
        }

        print(graph_info)

        return graph, graph_info
