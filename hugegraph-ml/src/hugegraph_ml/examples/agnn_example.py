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

from hugegraph_ml.data.hugegraph2dgl import HugeGraph2DGL
from hugegraph_ml.models.agnn import AGNN
from hugegraph_ml.tasks.node_classify import NodeClassify


def agnn_example(n_epochs=200):
    hg2d = HugeGraph2DGL()
    graph = hg2d.convert_graph(vertex_label="CORA_vertex", edge_label="CORA_edge")
    model = AGNN(
        num_layers=2,
        in_dim=graph.ndata["feat"].shape[1],
        hid_dim=64,
        out_dim=graph.ndata["label"].unique().shape[0],
        dropout=0.2,
    )
    node_clf_task = NodeClassify(graph, model)
    node_clf_task.train(lr=0.005, weight_decay=0.0005, n_epochs=n_epochs, patience=200)
    print(node_clf_task.evaluate())


if __name__ == "__main__":
    agnn_example()
