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

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

from hugegraph_ml.data.hugegraph2dgl import HugeGraph2DGL
from hugegraph_ml.models.grace import GRACE
from hugegraph_ml.tasks.node_embed import NodeEmbed


def grace_example(
        dataset_name="cora",
        lr=5e-4,
        n_epochs=200,
        wd=1e-5
):
    g2d = HugeGraph2DGL()
    graph, graph_info = g2d.convert_graph(
        vertex_label="cora_vertex", edge_label="cora_edge", info_vertex_label="cora_info_vertex"
    )
    model = GRACE(n_in_feats=graph_info["n_feat_dim"])
    node_embed_task = NodeEmbed(graph=graph, graph_info=graph_info, model=model)
    embedded_graph, graph_info = node_embed_task.train_and_embed(
        add_self_loop=True, lr=lr, weight_decay=wd, n_epochs=n_epochs, gpu=0
    )

    embeds = embedded_graph.ndata["feat"]
    labels = embedded_graph.ndata["label"]
    acc = classification(embeds, labels)
    print("GRACE: dataset {} test accuracy {:.4f}".format(dataset_name, acc))


def classification(embeddings, y):
    X = normalize(embeddings.detach().cpu().numpy(), norm="l2")
    Y = y.detach().cpu().numpy().reshape(-1, 1)
    Y = OneHotEncoder(categories="auto").fit_transform(Y).toarray().astype(np.bool_)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9)

    clf = GridSearchCV(estimator=OneVsRestClassifier(LogisticRegression(solver="liblinear")),
                       param_grid=dict(estimator__C=2.0 ** np.arange(-10, 10)),
                       n_jobs=8, cv=5, verbose=0)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)
    y_pred = np.eye(y_prob.shape[1])[np.argmax(y_prob, axis=1)].astype(np.bool_)
    acc = accuracy_score(y_test, y_pred)
    return acc

if __name__ == '__main__':
    grace_example()
