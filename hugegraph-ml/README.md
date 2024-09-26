# HugeGraph -ai-graph-learning-B

### 0. Environment

- python 3.9+

- hugegraph-server 1.5.0

### 1. Create new graph and init HugeGraph Database

#### 1.1. Create a new graph in HugeGraph

```
python utils/post_graph.py
```

#### 1.2. Init schema and data from edge list

```shell
python data_process/diy_hugegraph_init.py
```

Note: The vertex only has one property: id. Features and labels will be generated randomly when converting HugeGraph data to DGL graph.

### 2. Run

#### 2.1. For example, training appnp

```
python end2end/appnp.py
```

## 3. Others

#### 3.1. Clear the graph

```
python utils/clear_graph.py
```

Note: Maybe you should modify the parameter

#### 3.2. Delete the graph

```
python utils/delete_graph.py
```

