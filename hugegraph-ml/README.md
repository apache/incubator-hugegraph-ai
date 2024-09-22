 # hugegraph-ml

## Summary

`hugegraph-ml` is a tool that integrates HugeGraph with popular graph learning libraries. 
It implements most graph learning algorithms, enabling users to perform end-to-end graph learning workflows directly from HugeGraph using `hugegraph-ml`. 
Graph data can be read directly from `HugeGraph` and used for tasks such as node embedding, node classification, and graph classification. 
The implemented algorithm models can be found in the [models](./src/hugegraph_ml/models) folder.


## Environment Requirements

- python 3.9+ 
- hugegraph-server 1.0+

## Preparation

1. Start the HugeGraph database, you can do it via Docker/[Binary packages](https://hugegraph.apache.org/docs/download/download/). 
Refer to [docker-link](https://hub.docker.com/r/hugegraph/hugegraph) & [deploy-doc](https://hugegraph.apache.org/docs/quickstart/hugegraph-server/#31-use-docker-container-convenient-for-testdev) for guidance
2. Clone this project
    ```bash
    git clone https://github.com/apache/incubator-hugegraph-ai.git
    ```
3. Install [hugegraph-python-client](../hugegraph-python-client) and [hugegraph_ml](../hugegraph-ml)
    ```bash
    cd ./incubator-hugegraph-ai # better to use virtualenv (source venv/bin/activate) 
    pip install ./hugegraph-python-client
    cd ./hugegraph-ml/
    pip install -e .
    ```
4. Enter the project directory
    ```bash
    cd ./hugegraph-ml/src
    ```

## Examples

### Perform node embedding on the `Cora` dataset using the `DGI` model

Make sure that the Cora dataset is already in your HugeGraph database. 
If not, you can run the `import_graph_from_dgl` function to import the `Cora` dataset from `DGL` into
the `HugeGraph` database.

```python
from hugegraph_ml.utils.dgl2hugegraph_utils import import_graph_from_dgl

import_graph_from_dgl("cora")
```

Run [dgi_example.py](./src/hugegraph_ml/examples/dgi_example.py) to view the example.
```bash
python ./hugegraph_ml/examples/dgi_example.py
```

The specific process is as follows:

**1. Graph data convert**

Convert the graph from `HugeGraph` to `DGL` format.

```python
from hugegraph_ml.data.hugegraph2dgl import HugeGraph2DGL
from hugegraph_ml.models.dgi import DGI
from hugegraph_ml.models.mlp import MLPClassifier
from hugegraph_ml.tasks.node_classify import NodeClassify
from hugegraph_ml.tasks.node_embed import NodeEmbed

hg2d = HugeGraph2DGL()
graph, graph_info = hg2d.convert_graph(
   info_vertex_label="cora_info_vertex", 
   vertex_label="cora_vertex",
   edge_label="cora_edge"
)
```

**2. Select model instance**

```python
model = DGI(n_in_feats=graph_info["n_feat_dim"])
```

**3. Train model and node embedding**

```python
node_embed_task = NodeEmbed(graph=graph, graph_info=graph_info, model=model)
embedded_graph, graph_info = node_embed_task.train_and_embed(add_self_loop=True, n_epochs=300, patience=30)
```

**4. Downstream tasks node classification using MLP**

```python
model = MLPClassifier(n_in_feat=graph_info["n_feat_dim"], n_out_feat=graph_info["n_classes"])
node_clf_task = NodeClassify(graph=embedded_graph, graph_info=graph_info, model=model)
node_clf_task.train(lr=1e-3, n_epochs=400, patience=40)
print(node_clf_task.evaluate())
```

**5. Obtain the metrics**

```text
{'accuracy': 0.82, 'loss': 0.5714246034622192}
```

### Perform node classification on the `Cora` dataset using the `GRAND` model.

You can refer to the example in the [grand_example.py](./src/hugegraph_ml/examples/grand_example.py)

```python
from hugegraph_ml.data.hugegraph2dgl import HugeGraph2DGL
from hugegraph_ml.models.grand import GRAND
from hugegraph_ml.tasks.node_classify import NodeClassify

hg2d = HugeGraph2DGL()
graph, graph_info = hg2d.convert_graph(
   info_vertex_label="cora_info_vertex", 
   vertex_label="cora_vertex",
   edge_label="cora_edge"
)
model = GRAND(
    n_in_feats=graph_info["n_feat_dim"],
    n_out_feats=graph_info["n_classes"]
)
node_clf_task = NodeClassify(graph, graph_info, model)
node_clf_task.train(lr=1e-2, weight_decay=5e-4, n_epochs=2000, patience=100)
print(node_clf_task.evaluate())
```
