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

from typing import Optional

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, LegacyTUDataset, GINDataset
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

    client: PyHugeClient = PyHugeClient(ip=ip, port=port, graph=graph, user=user, pwd=pwd, graphspace=graphspace)
    client_schema: SchemaManager = client.schema()
    client_graph: GraphManager = client.graph()

    graph_dgl = dataset_dgl[0]
    node_features = graph_dgl.ndata["feat"]
    node_labels = graph_dgl.ndata["label"]
    edges_src, edges_dst = graph_dgl.edges()

    client_schema.propertyKey("feat").asDouble().valueList().ifNotExist().create()
    client_schema.propertyKey("label").asLong().ifNotExist().create()

    vertex_label = f"{dataset_name}_vertex"
    edge_label = f"{dataset_name}_edge"
    client_schema.vertexLabel(vertex_label).useCustomizeNumberId().properties("feat", "label").ifNotExist().create()
    client_schema.edgeLabel(edge_label).sourceLabel(vertex_label).targetLabel(vertex_label).ifNotExist().create()

    for node_id in range(graph_dgl.number_of_nodes()):
        node_feature = node_features[node_id].tolist()
        node_label = int(node_labels[node_id])
        client_graph.addVertex(label=vertex_label, properties={"feat": node_feature, "label": node_label}, id=node_id)

    for src, dst in zip(edges_src.numpy(), edges_dst.numpy()):
        client_graph.addEdge(edge_label=edge_label, out_id=src.item(), in_id=dst.item(), properties={})

    client_schema.propertyKey("train_mask").asInt().valueList().ifNotExist().create()
    client_schema.propertyKey("val_mask").asLong().valueList().ifNotExist().create()
    client_schema.propertyKey("test_mask").asLong().valueList().ifNotExist().create()
    info_vertex = f"{dataset_name}_graph_vertex"
    client_schema.vertexLabel(info_vertex).useAutomaticId().properties(
        "train_mask", "val_mask", "test_mask"
    ).ifNotExist().create()

    train_mask = graph_dgl.ndata["train_mask"].int()
    val_mask = graph_dgl.ndata["val_mask"].int()
    test_mask = graph_dgl.ndata["test_mask"].int()
    client_graph.addVertex(
        label=info_vertex,
        properties={"train_mask": train_mask.tolist(), "val_mask": val_mask.tolist(), "test_mask": test_mask.tolist()}
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
        assert graph_dgl.number_of_nodes() == node_feats.shape[0]
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
    client_graph.close()

if __name__ == "__main__":
    clear_all_data()
    import_graph_from_dgl("cora")
    import_graphs_from_dgl("MUTAG")
