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

from ..client_utils import ClientUtils


class TestSchemaManager(unittest.TestCase):
    client = None
    schema = None

    @classmethod
    def setUpClass(cls):
        cls.client = ClientUtils()
        cls.client.clear_graph_all_data()
        cls.schema = cls.client.schema
        cls.client.init_property_key()
        cls.client.init_vertex_label()
        cls.client.init_edge_label()
        cls.client.init_index_label()

    @classmethod
    def tearDownClass(cls):
        cls.client.clear_graph_all_data()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_schema(self):
        schema = self.schema.getSchema()
        self.assertEqual(4, len(schema))

    def test_get_property_keys(self):
        property_keys = self.schema.getPropertyKeys()
        self.assertEqual(7, len(property_keys))

    def test_get_property_key(self):
        property_key = self.schema.getPropertyKey("name")
        self.assertEqual(property_key.name, "name")

    def test_get_vertex_labels(self):
        vertex_labels = self.schema.getVertexLabels()
        self.assertEqual(3, len(vertex_labels))

    def test_get_vertex_label(self):
        vertex_label = self.schema.getVertexLabel("person")
        self.assertEqual(vertex_label.name, "person")

    def test_get_edge_labels(self):
        edge_labels = self.schema.getEdgeLabels()
        self.assertEqual(2, len(edge_labels))

    def test_get_edge_label(self):
        edge_label = self.schema.getEdgeLabel("knows")
        self.assertEqual(edge_label.name, "knows")

    def test_get_index_labels(self):
        index_labels = self.schema.getIndexLabels()
        self.assertEqual(6, len(index_labels))

    def test_get_index_label(self):
        index_label = self.schema.getIndexLabel("personByCity")
        self.assertEqual(index_label.name, "personByCity")
