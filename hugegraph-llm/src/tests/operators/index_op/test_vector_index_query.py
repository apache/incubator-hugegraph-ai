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

# pylint: disable=unused-argument

import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.operators.index_op.vector_index_query import VectorIndexQuery


class MockEmbedding(BaseEmbedding):
    """Mock embedding class for testing"""

    def __init__(self):
        super().__init__()  # Call parent class constructor
        self.model = "mock_model"

    def get_text_embedding(self, text):
        # Return a simple mock embedding based on the text
        if text == "query1":
            return [1.0, 0.0, 0.0, 0.0]
        if text == "query2":
            return [0.0, 1.0, 0.0, 0.0]
        return [0.5, 0.5, 0.0, 0.0]

    def get_texts_embeddings(self, texts):
        # Return embeddings for multiple texts
        return [self.get_text_embedding(text) for text in texts]

    async def async_get_text_embedding(self, text):
        # Async version returns the same as the sync version
        return self.get_text_embedding(text)

    async def async_get_texts_embeddings(self, texts):
        # Async version of get_texts_embeddings
        return [await self.async_get_text_embedding(text) for text in texts]

    def get_llm_type(self):
        return "mock"


class TestVectorIndexQuery(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

        # Create a mock embedding model
        self.embedding = MockEmbedding()

        # Create sample vectors and properties for the index
        self.embed_dim = 4
        self.vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        self.properties = ["doc1", "doc2", "doc3", "doc4"]

        # Create a mock vector index
        self.mock_index = MagicMock()
        self.mock_index.search.return_value = ["doc1"]  # Default return value

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    @patch("hugegraph_llm.operators.index_op.vector_index_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.vector_index_query.resource_path")
    @patch("hugegraph_llm.operators.index_op.vector_index_query.huge_settings")
    def test_init(self, mock_settings, mock_resource_path, mock_vector_index_class):
        # Configure mocks
        mock_settings.graph_name = "test_graph"
        mock_resource_path = "/mock/path"
        mock_vector_index_class.from_index_file.return_value = self.mock_index

        # Create a VectorIndexQuery instance
        with patch("os.path.join", return_value=self.test_dir):
            query = VectorIndexQuery(self.embedding, topk=3)

            # Verify the instance was initialized correctly
            self.assertEqual(query.embedding, self.embedding)
            self.assertEqual(query.topk, 3)
            self.assertEqual(query.vector_index, self.mock_index)
            mock_vector_index_class.from_index_file.assert_called_once()

    @patch("hugegraph_llm.operators.index_op.vector_index_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.vector_index_query.resource_path")
    @patch("hugegraph_llm.operators.index_op.vector_index_query.huge_settings")
    def test_run(self, mock_settings, mock_resource_path, mock_vector_index_class):
        # Configure mocks
        mock_settings.graph_name = "test_graph"
        mock_resource_path = "/mock/path"
        mock_vector_index_class.from_index_file.return_value = self.mock_index
        self.mock_index.search.return_value = ["doc1"]

        # Create a context with a query
        context = {"query": "query1"}

        # Create a VectorIndexQuery instance
        with patch("os.path.join", return_value=self.test_dir):
            query = VectorIndexQuery(self.embedding, topk=2)

            # Run the query
            result_context = query.run(context)

            # Verify the results
            self.assertIn("vector_result", result_context)
            self.assertEqual(result_context["vector_result"], ["doc1"])

            # Verify the mock was called correctly
            self.mock_index.search.assert_called_once()
            # First argument should be the embedding for "query1"
            args, _ = self.mock_index.search.call_args
            self.assertEqual(args[0], [1.0, 0.0, 0.0, 0.0])

    @patch("hugegraph_llm.operators.index_op.vector_index_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.vector_index_query.resource_path")
    @patch("hugegraph_llm.operators.index_op.vector_index_query.huge_settings")
    def test_run_with_different_query(
        self, mock_settings, mock_resource_path, mock_vector_index_class
    ):
        # Configure mocks
        mock_settings.graph_name = "test_graph"
        mock_resource_path = "/mock/path"
        mock_vector_index_class.from_index_file.return_value = self.mock_index
        self.mock_index.search.return_value = ["doc2"]

        # Create a context with a different query
        context = {"query": "query2"}

        # Create a VectorIndexQuery instance
        with patch("os.path.join", return_value=self.test_dir):
            query = VectorIndexQuery(self.embedding, topk=2)

            # Run the query
            result_context = query.run(context)

            # Verify the results
            self.assertIn("vector_result", result_context)
            self.assertEqual(result_context["vector_result"], ["doc2"])

            # Verify the mock was called correctly
            self.mock_index.search.assert_called_once()
            # First argument should be the embedding for "query2"
            args, _ = self.mock_index.search.call_args
            self.assertEqual(args[0], [0.0, 1.0, 0.0, 0.0])

    @patch("hugegraph_llm.operators.index_op.vector_index_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.vector_index_query.resource_path")
    @patch("hugegraph_llm.operators.index_op.vector_index_query.huge_settings")
    def test_run_with_empty_context(
        self, mock_settings, mock_resource_path, mock_vector_index_class
    ):
        # Configure mocks
        mock_settings.graph_name = "test_graph"
        mock_resource_path = "/mock/path"
        mock_vector_index_class.from_index_file.return_value = self.mock_index

        # Create an empty context
        context = {}

        # Create a VectorIndexQuery instance
        with patch("os.path.join", return_value=self.test_dir):
            query = VectorIndexQuery(self.embedding, topk=2)

            # Run the query with empty context
            result_context = query.run(context)

            # Verify the results
            self.assertIn("vector_result", result_context)

            # Verify the mock was called with the default embedding
            self.mock_index.search.assert_called_once()
            args, _ = self.mock_index.search.call_args
            self.assertEqual(args[0], [0.5, 0.5, 0.0, 0.0])  # Default embedding for None
