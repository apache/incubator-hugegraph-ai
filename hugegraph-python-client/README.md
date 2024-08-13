# hugegraph-python

A Python SDK for Apache HugeGraph

## Installation

```shell
pip3 install hugegraph-python
```

### Install from source

```bash
cd /path/to/hugegraph-python-client

# install 
pip install .

# If you want to install the devel version
pip install -e .
```

## Examples

```python
from pyhugegraph.client import PyHugeClient

# For users of HugeGraph API version 3 and higher, the 'gs' parameter becomes relevant if graph spaces are enabled.
# It specifies the name of the graph space to operate on. If graph spaces are not enabled, this parameter is optional and can be omitted.
# The default graph space name is often 'DEFAULT', but it depends on your HugeGraph server configuration.
# If you're using a version older than v3 or if graph spaces are not enabled, simply omit the 'gs' parameter.
client = PyHugeClient("127.0.0.1", "8080", user="admin", pwd="admin", graph="hugegraph", gs="DEFAULT") # For v3+ with graph spaces enabled. Omit 'gs' for older versions or without graph spaces.

"""
Note:

Starting with HugeGraph API version 3, the gs parameter is necessary when graph spaces are enabled to specify the graph space context within which operations are performed.
If you’re using a version older than v3 or your HugeGraph instance does not have graph spaces enabled, the gs parameter is optional and you can safely omit it from the PyHugeClient initialization.
Always refer to the documentation of your HugeGraph version for accurate configuration details and to ensure compatibility with the features you intend to use.
"""

"""system"""
print(client.get_graphinfo())
print(client.get_all_graphs())
print(client.get_version())
print(client.get_graph_config())

"""schema"""
schema = client.schema()
schema.propertyKey("name").asText().ifNotExist().create()
schema.propertyKey("birthDate").asText().ifNotExist().create()
schema.vertexLabel("Person").properties("name", "birthDate").usePrimaryKeyId().primaryKeys(
    "name").ifNotExist().create()
schema.vertexLabel("Movie").properties("name").usePrimaryKeyId().primaryKeys(
    "name").ifNotExist().create()
schema.edgeLabel("ActedIn").sourceLabel("Person").targetLabel("Movie").ifNotExist().create()

print(schema.getVertexLabels())
print(schema.getEdgeLabels())
print(schema.getRelations())

"""graph"""
g = client.graph()
g.addVertex("Person", {"name": "Al Pacino", "birthDate": "1940-04-25"})
g.addVertex("Person", {"name": "Robert De Niro", "birthDate": "1943-08-17"})
g.addVertex("Movie", {"name": "The Godfather"})
g.addVertex("Movie", {"name": "The Godfather Part II"})
g.addVertex("Movie", {"name": "The Godfather Coda The Death of Michael Corleone"})

g.addEdge("ActedIn", "12:Al Pacino", "13:The Godfather", {})
g.addEdge("ActedIn", "12:Al Pacino", "13:The Godfather Part II", {})
g.addEdge("ActedIn", "12:Al Pacino", "13:The Godfather Coda The Death of Michael Corleone", {})
g.addEdge("ActedIn", "12:Robert De Niro", "13:The Godfather Part II", {})

res = g.getVertexById("12:Al Pacino").label
# g.removeVertexById("12:Al Pacino")
print(res)
g.close()

"""gremlin"""
g = client.gremlin()
res = g.exec("g.V().limit(10)")
print(res)
```
