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

import unittest

from pyhugegraph.utils.exceptions import NotFoundError
from tests.client_utils import ClientUtils


class TestGraphManager(unittest.TestCase):
    client = None
    graph = None

    @classmethod
    def setUpClass(cls):
        cls.client = ClientUtils()
        cls.graph = cls.client.graph
        cls.client.init_property_key()
        cls.client.init_vertex_label()
        cls.client.init_edge_label()
        cls.client.init_index_label()

    @classmethod
    def tearDownClass(cls):
        cls.client.clear_graph_all_data()

    def test_add_vertex(self):
        vertex = self.graph.addVertex("person", {"name": "marko", "age": 29, "city": "Beijing"})
        self.assertIsNotNone(vertex)

    def test_add_vertices(self):
        vertices = self.graph.addVertices(
            [
                ("person", {"name": "vadas", "age": 27, "city": "Hongkong"}),
                ("person", {"name": "marko", "age": 29, "city": "Beijing"}),
            ]
        )
        self.assertEqual(len(vertices), 2)

    def test_append_vertex(self):
        vertex = self.graph.addVertex("person", {"name": "Alice", "age": 20})
        appended_vertex = self.graph.appendVertex(vertex.id, {"city": "Beijing"})
        self.assertEqual(appended_vertex.properties["city"], "Beijing")

    def test_eliminate_vertex(self):
        vertex = self.graph.addVertex("person", {"name": "marko", "age": 29, "city": "Beijing"})
        self.graph.eliminateVertex(vertex.id, {"city": "Beijing"})
        eliminated_vertex = self.graph.getVertexById(vertex.id)
        self.assertIsNone(eliminated_vertex.properties.get("city"))

    def test_get_vertex_by_id(self):
        vertex = self.graph.addVertex("person", {"name": "Alice", "age": 20})
        retrieved_vertex = self.graph.getVertexById(vertex.id)
        self.assertEqual(retrieved_vertex.id, vertex.id)

    def test_get_vertex_by_page(self):
        self.graph.addVertex("person", {"name": "Alice", "age": 20})
        self.graph.addVertex("person", {"name": "Bob", "age": 23})
        vertices = self.graph.getVertexByPage("person", 1)
        self.assertEqual(len(vertices), 2)

    def test_get_vertex_by_condition(self):
        self.graph.addVertex("person", {"name": "Alice", "age": 25})
        self.graph.addVertex("person", {"name": "Bob", "age": 30})
        vertices = self.graph.getVertexByCondition("person", properties={"age": "P.gt(29)"})
        self.assertEqual(len(vertices), 1)
        self.assertEqual(vertices[0].properties["name"], "Bob")

    def test_remove_vertex_by_id(self):
        vertex = self.graph.addVertex("person", {"name": "Alice", "age": 20})
        self.graph.removeVertexById(vertex.id)
        try:
            self.graph.getVertexById(vertex.id)
        except NotFoundError as e:
            self.assertTrue("Alice\\' does not exist" in str(e))

    def test_add_edge(self):
        vertex1 = self.graph.addVertex("person", {"name": "Alice", "age": 20})
        vertex2 = self.graph.addVertex("person", {"name": "Bob", "age": 23})
        edge = self.graph.addEdge("knows", vertex1.id, vertex2.id, {"date": "2012-01-10"})
        self.assertIsNotNone(edge)

    def test_add_edges(self):
        vertex1 = self.graph.addVertex("person", {"name": "Alice", "age": 20})
        vertex2 = self.graph.addVertex("person", {"name": "Bob", "age": 23})

        vertices = self.graph.addEdges(
            [
                ("knows", vertex1.id, vertex2.id, "person", "person", {"date": "2012-01-10"}),
                ("knows", vertex2.id, vertex1.id, "person", "person", {"date": "2012-01-10"}),
            ]
        )
        self.assertEqual(len(vertices), 2)

    def test_append_edge(self):
        vertex1 = self.graph.addVertex("person", {"name": "Alice", "age": 20})
        vertex2 = self.graph.addVertex("person", {"name": "Bob", "age": 23})
        edge = self.graph.addEdge("knows", vertex1.id, vertex2.id, {"date": "2012-01-10"})
        appended_edge = self.graph.appendEdge(edge.id, {"city": "Beijing"})
        self.assertEqual(appended_edge.properties["city"], "Beijing")

    def test_eliminate_edge(self):
        vertex1 = self.graph.addVertex("person", {"name": "Alice", "age": 20})
        vertex2 = self.graph.addVertex("person", {"name": "Bob", "age": 23})
        edge = self.graph.addEdge("knows", vertex1.id, vertex2.id, {"date": "2012-01-10"})
        self.graph.eliminateEdge(edge.id, {"city": "Beijing"})
        eliminated_edge = self.graph.getEdgeById(edge.id)
        self.assertIsNone(eliminated_edge.properties.get("city"))

    def test_get_edge_by_id(self):
        vertex1 = self.graph.addVertex("person", {"name": "Alice", "age": 20})
        vertex2 = self.graph.addVertex("person", {"name": "Bob", "age": 23})
        edge = self.graph.addEdge("knows", vertex1.id, vertex2.id, {"date": "2012-01-10"})
        retrieved_edge = self.graph.getEdgeById(edge.id)
        self.assertEqual(retrieved_edge.id, edge.id)

    def test_get_edge_by_page(self):
        vertex1 = self.graph.addVertex("person", {"name": "Alice", "age": 20})
        vertex2 = self.graph.addVertex("person", {"name": "Bob", "age": 23})
        self.graph.addEdge("knows", vertex1.id, vertex2.id, {"date": "2012-01-10"})
        self.graph.addEdge("knows", vertex2.id, vertex1.id, {"date": "2012-01-10"})
        edges = self.graph.getEdgeByPage("knows")
        self.assertEqual(len(edges), 2)

    def test_remove_edge_by_id(self):
        vertex1 = self.graph.addVertex("person", {"name": "Alice", "age": 20})
        vertex2 = self.graph.addVertex("person", {"name": "Bob", "age": 23})
        edge = self.graph.addEdge("knows", vertex1.id, vertex2.id, {"date": "2012-01-10"})
        self.graph.removeEdgeById(edge.id)
        try:
            self.graph.getEdgeById(edge.id)
        except NotFoundError as e:
            self.assertTrue("does not exist" in str(e))

    def test_get_vertices_by_id(self):
        vertex1 = self.graph.addVertex("person", {"name": "Alice", "age": 20})
        vertex2 = self.graph.addVertex("person", {"name": "Bob", "age": 23})
        vertices = self.graph.getVerticesById([vertex1.id, vertex2.id])
        self.assertEqual(len(vertices), 2)

    def test_get_edges_by_id(self):
        vertex1 = self.graph.addVertex("person", {"name": "Alice", "age": 20})
        vertex2 = self.graph.addVertex("person", {"name": "Bob", "age": 23})
        edge1 = self.graph.addEdge("knows", vertex1.id, vertex2.id, {"date": "2012-01-10"})
        edge2 = self.graph.addEdge("knows", vertex2.id, vertex1.id, {"date": "2012-01-10"})
        edges = self.graph.getEdgesById([edge1.id, edge2.id])
        self.assertEqual(len(edges), 2)
