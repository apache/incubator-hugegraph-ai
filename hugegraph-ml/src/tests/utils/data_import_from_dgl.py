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


from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from pyhugegraph.api.graph import GraphManager
from pyhugegraph.api.schema import SchemaManager
from pyhugegraph.client import PyHugeClient


def clear_all_data():
    client: PyHugeClient = PyHugeClient(
        ip='127.0.0.1',
        port="8080",
        graph='hugegraph',
        user='admin',
        pwd='xxx'
    )
    client.graphs().clear_graph_all_data()

def import_graph_from_dgl(dataset_name):
    if dataset_name == 'cora':
        dataset_dgl =CoraGraphDataset()
    elif dataset_name == 'citeseer':
        dataset_dgl =CiteseerGraphDataset()
    elif dataset_name == 'pubmed':
        dataset_dgl =PubmedGraphDataset()
    else:
        raise ValueError('dataset not supported')


    client: PyHugeClient = PyHugeClient(
        ip='127.0.0.1',
        port="8080",
        graph='hugegraph',
        user='admin',
        pwd='xxx'
    )
    schema: SchemaManager = client.schema()
    graph: GraphManager = client.graph()


    graph_dgl = dataset_dgl[0]
    node_features = graph_dgl.ndata["feat"]
    node_labels = graph_dgl.ndata["label"]
    edges_src, edges_dst = graph_dgl.edges()

    print(graph_dgl.ndata['feat'].shape)

    schema.propertyKey("feat").asDouble().valueList().ifNotExist().create()
    schema.propertyKey("label").asLong().ifNotExist().create()

    vertex_label = f"{dataset_name}_vertex"
    edge_label = f"{dataset_name}_edge"
    schema.vertexLabel(vertex_label).useCustomizeStringId().properties("feat", "label").ifNotExist().create()
    schema.edgeLabel(edge_label).sourceLabel(vertex_label).targetLabel(vertex_label).ifNotExist().create()


    for node_id in range(graph_dgl.number_of_nodes()):
        node_feature = node_features[node_id].tolist()
        node_label = int(node_labels[node_id])
        graph.addVertex(
            label=vertex_label,
            properties={
                "feat": node_feature,
                "label": node_label
            },
            id=dataset_name + str(node_id)
        )

    for src, dst in zip(edges_src.numpy(), edges_dst.numpy()):
        graph.addEdge(
            edge_label=edge_label,
            out_id=dataset_name + str(src.item()),
            in_id=dataset_name + str(dst.item()),
            properties={}
        )

    schema.propertyKey("train_mask").asInt().valueList().ifNotExist().create()
    schema.propertyKey("val_mask").asLong().valueList().ifNotExist().create()
    schema.propertyKey("test_mask").asLong().valueList().ifNotExist().create()
    info_vertex = f"{dataset_name}_info_vertex"
    schema.vertexLabel(info_vertex).useCustomizeStringId().properties("train_mask", "val_mask", "test_mask").ifNotExist().create()

    train_mask = graph_dgl.ndata["train_mask"].int()
    val_mask = graph_dgl.ndata["val_mask"].int()
    test_mask = graph_dgl.ndata["test_mask"].int()
    graph.addVertex(
        label=info_vertex,
        properties={
            "train_mask": train_mask.tolist(),
            "val_mask": val_mask.tolist(),
            "test_mask": test_mask.tolist()
        },
        id=dataset_name + "_info"
    )



if __name__ == "__main__":
    clear_all_data()
    import_graph_from_dgl('cora')
