# HugeGraph -ai-graph-learning-B

### 0. Code Tree

- `data_process`
  - `diy_hugegraph_init.py`  init the schema and data from **edge list**
  - `diy_hugegraph2dgl.py`   combline APIs that generate features and labels randomly and split the dataset
- `dataset`
  - `cora`
    - `cora.cites` the edge list of cora
- `end2end`  run the models
  - `appnp.py`
  - `arma.py`
  - `cluster_gcn.py`
  - `dagnn.py`
- `models`
  - `appnp.py`
  - `arma.py`
  - `cluster_gcn.py`
  - `dagnn.py`
- `utils`
  - `clear_graph.py` clear a HugeGraph graph
  - `delete_graph.py` delete a HugeGraph graph
  - `post_graph.py` create a new HugeGraph graph

### 1. Environment

- python 3.9+

- hugegraph-server 1.5.0

### 2. Create new graph and init HugeGraph Database

#### 2.1. Create a new graph in HugeGraph

```
python utils/post_graph.py
```

#### 2.2. Init schema and data from edge list

```shell
python data_process/diy_hugegraph_init.py
```

Note: The vertex only has one property: id. Features and labels will be generated randomly when converting HugeGraph data to DGL graph.

### 3. Run

#### 3.1. For example, training appnp

```
python end2end/appnp.py
```

## 4. Others

#### 4.1. Clear the graph

```
python utils/clear_graph.py
```

Note: Maybe you should modify the parameter

#### 4.2. Delete the graph

```
python utils/delete_graph.py
```

