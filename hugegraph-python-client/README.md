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

# The 'gs' parameter is optional and represents the name of the graph space to operate on (if graph spaces are configured in your HugeGraph setup).
# The default is usually 'DEFAULT' (depending on your HugeGraph server configuration).
# If your HugeGraph configuration enables graph spaces and you need to operate on a specific graph space, you can specify it here.
client = PyHugeClient("127.0.0.1", "8080", user="admin", pwd="admin", graph="hugegraph", gs="DEFAULT") # If graph spaces are not enabled or not of concern, the 'gs' parameter can be omitted

"""
Note:

The ‘gs’ parameter is used to specify the graph space to operate on. In graph database management, graph spaces are often used to separate and manage different graph structures within a single HugeGraph cluster, providing isolation and resource management capabilities. If your use case involves multiple graphs or graph spaces, you need to specify the ‘gs’ parameter.
The ‘graph’ parameter refers to the specific graph you want to operate on, not an alias. There can be multiple graphs within the same graph space.
Make sure you modify the IP address, port number, authentication information, etc., in the above sample code according to your HugeGraph configuration and service status.
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
