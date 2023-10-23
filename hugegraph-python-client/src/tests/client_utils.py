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

from pyhugegraph.client import PyHugeClient


class ClientUtils:
    IP = "127.0.0.1"
    PORT = 8080
    GRAPH = "hugegraph"
    USERNAME = 'admin'
    PASSWORD = 'admin'
    TIMEOUT = 10

    def __init__(self):
        self.client = PyHugeClient(self.IP, self.PORT, user=self.USERNAME,
                                   pwd=self.PASSWORD, graph=self.GRAPH)
        assert self.client is not None

        self.schema = self.client.schema()
        self.gremlin = self.client.gremlin()
        self.graph = self.client.graph()
        self.graphs = self.client.graphs()
        self.variable = self.client.variable()

    def init_property_key(self):
        schema = self.schema
        schema.propertyKey("name").asText().ifNotExist().create()
        schema.propertyKey("age").asInt().ifNotExist().create()
        schema.propertyKey("city").asText().ifNotExist().create()
        schema.propertyKey("lang").asText().ifNotExist().create()
        schema.propertyKey("date").asDate().ifNotExist().create()
        schema.propertyKey("price").asInt().ifNotExist().create()
        schema.propertyKey("weight").asDouble().ifNotExist().create()

    def init_vertex_label(self):
        schema = self.schema
        schema.vertexLabel("person").properties("name", "age", "city").primaryKeys("name") \
            .nullableKeys("city").ifNotExist().create()
        schema.vertexLabel("software").properties("name", "lang", "price").primaryKeys("name") \
            .nullableKeys("price").ifNotExist().create()
        schema.vertexLabel("book").useCustomizeStringId().properties("name", "price") \
            .nullableKeys("price").ifNotExist().create()

    def init_edge_label(self):
        schema = self.schema
        schema.edgeLabel("knows").sourceLabel("person").targetLabel("person") \
            .multiTimes().properties("date", "city").sortKeys("date") \
            .nullableKeys("city").ifNotExist().create()
        schema.edgeLabel("created").sourceLabel("person").targetLabel("software") \
            .properties("date", "city").nullableKeys("city").ifNotExist().create()

    def init_vertices(self):
        graph = self.graph
        graph.addVertex("person", {"name": "marko", "age": 29, "city": "Beijing"})
        graph.addVertex("person", {"name": "vadas", "age": 27, "city": "Hongkong"})
        graph.addVertex("software", {"name": "lop", "lang": "java", "price": 328})
        graph.addVertex("person", {"name": "josh", "age": 32, "city": "Beijing"})
        graph.addVertex("software", {"name": "ripple", "lang": "java", "price": 199})
        graph.addVertex("person", {"name": "peter", "age": 29, "city": "Shanghai"})

    def init_edges(self):
        marko_id = self._get_vertex_id("person", {"name": "marko"})
        vadas_id = self._get_vertex_id("person", {"name": "vadas"})
        josh_id = self._get_vertex_id("person", {"name": "josh"})
        peter_id = self._get_vertex_id("person", {"name": "peter"})
        lop_id = self._get_vertex_id("software", {"name": "lop"})
        ripple_id = self._get_vertex_id("software", {"name": "ripple"})

        self.graph.addEdge("knows", marko_id, vadas_id, {"date": "2012-01-10"})
        self.graph.addEdge("knows", marko_id, josh_id, {"date": "2013-01-10"})
        self.graph.addEdge("created", marko_id, lop_id, {"date": "2014-01-10", "city": "Shanghai"})
        self.graph.addEdge("created", josh_id, ripple_id, {"date": "2015-01-10", "city": "Beijing"})
        self.graph.addEdge("created", josh_id, lop_id, {"date": "2016-01-10", "city": "Beijing"})
        self.graph.addEdge("created", peter_id, lop_id, {"date": "2017-01-10", "city": "Hongkong"})

    def _get_vertex_id(self, label, properties):
        res = self._get_vertex(label, properties)
        return res.id

    def _get_vertex(self, label, properties):
        lst = self.graph.getVertexByCondition(label=label, limit=1, properties=properties)
        assert 1 == len(lst), "Can't find vertex."
        return lst[0]

    def clear_graph_all_data(self):
        self.graphs.clear_graph_all_data()
