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

import pytest

from pyhugegraph.utils.exceptions import NotFoundError
from tests.client_utils import ClientUtils


class TestGremlin(unittest.TestCase):
    client = None
    gremlin = None

    @classmethod
    def setUpClass(cls):
        cls.client = ClientUtils()
        cls.client.clear_graph_all_data()
        cls.gremlin = cls.client.gremlin
        cls.client.init_property_key()
        cls.client.init_vertex_label()
        cls.client.init_edge_label()

    @classmethod
    def tearDownClass(cls):
        cls.client.clear_graph_all_data()

    def setUp(self):
        self.client.init_vertices()
        self.client.init_edges()

    def tearDown(self):
        pass

    def test_query_all_vertices(self):
        vertices = self.gremlin.exec("g.V()")
        lst = vertices.get("data", [])
        assert 6 == len(lst)

        self.gremlin.exec("g.V().drop()")
        vertices = self.gremlin.exec("g.V()")
        lst = vertices.get("data", [])
        assert 0 == len(lst)

    def test_query_all_edges(self):
        edges = self.gremlin.exec("g.E()")
        lst = edges.get("data", [])
        assert 6 == len(lst)

        self.gremlin.exec("g.E().drop()")
        edges = self.gremlin.exec("g.E()")
        lst = edges.get("data", [])
        assert 0 == len(lst)

    def test_primitive_object(self):
        result = self.gremlin.exec("1 + 2")
        print(result)
        result_set = result.get("data", [])
        assert 1 == len(result_set)

        data = result_set[0]
        assert isinstance(data, int)
        assert 3 == data

    def test_empty_result_set(self):
        result = self.gremlin.exec("g.V().limit(0)")
        lst = result.get("data", [])
        assert 0 == len(lst)

    def test_invalid_gremlin(self):
        with pytest.raises(NotFoundError):
            assert self.gremlin.exec("g.V2()")

    def test_security_operation(self):
        with pytest.raises(NotFoundError):
            assert self.gremlin.exec("System.exit(-1)")
