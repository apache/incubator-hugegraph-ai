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
from hugegraph_ml.models.grace import GRACE
from hugegraph_ml.models.mlp import MLPClassifier
from hugegraph_ml.tasks.node_classify import NodeClassify
from hugegraph_ml.tasks.node_embed import NodeEmbed


def grace_example():
    hg2d = HugeGraph2DGL()
    graph, graph_info = hg2d.convert_graph(
        vertex_label="cora_vertex", edge_label="cora_edge", info_vertex_label="cora_info_vertex"
    )
    model = GRACE(n_in_feats=graph_info["n_feat_dim"])
    node_embed_task = NodeEmbed(graph=graph, graph_info=graph_info, model=model)
    embedded_graph, graph_info = node_embed_task.train_and_embed(
        add_self_loop=True, lr=0.001, weight_decay=1e-5, n_epochs=400, patience=40
    )
    model = MLPClassifier(
        n_in_feat=graph_info["n_feat_dim"],
        n_out_feat=graph_info["n_classes"],
        n_hidden=128
    )
    node_clf_task = NodeClassify(graph=embedded_graph, graph_info=graph_info, model=model)
    node_clf_task.train(lr=1e-3, n_epochs=300, patience=30)
    print(node_clf_task.evaluate())


if __name__ == '__main__':
    grace_example()
