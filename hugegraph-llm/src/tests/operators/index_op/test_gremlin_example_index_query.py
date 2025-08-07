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

# pylint: disable=unused-argument,unused-variable

import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.operators.index_op.gremlin_example_index_query import GremlinExampleIndexQuery


class MockEmbedding(BaseEmbedding):
    """Mock embedding class for testing"""

    def __init__(self):
        self.model = "mock_model"

    def get_text_embedding(self, text):
        # Return a simple mock embedding based on the text
        if text == "find all persons":
            return [1.0, 0.0, 0.0, 0.0]
        if text == "count movies":
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


class TestGremlinExampleIndexQuery(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

        # Create a mock embedding model
        self.embedding = MockEmbedding()

        # Create sample vectors and properties for the index
        self.embed_dim = 4
        self.vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        self.properties = [
            {"query": "find all persons", "gremlin": "g.V().hasLabel('person')"},
            {"query": "count movies", "gremlin": "g.V().hasLabel('movie').count()"},
        ]

        # Create a mock vector index
        self.mock_index = MagicMock()
        self.mock_index.search.return_value = [self.properties[0]]  # Default return value

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    @patch("hugegraph_llm.operators.index_op.gremlin_example_index_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.gremlin_example_index_query.resource_path")
    def test_init(self, mock_resource_path, mock_vector_index_class):
        # Configure mocks
        mock_resource_path = "/mock/path"
        mock_vector_index_class.from_index_file.return_value = self.mock_index

        # Create a GremlinExampleIndexQuery instance
        with patch("os.path.join", return_value=self.test_dir):
            query = GremlinExampleIndexQuery(self.embedding, num_examples=2)

            # Verify the instance was initialized correctly
            self.assertEqual(query.embedding, self.embedding)
            self.assertEqual(query.num_examples, 2)
            self.assertEqual(query.vector_index, self.mock_index)
            mock_vector_index_class.from_index_file.assert_called_once()

    @patch("hugegraph_llm.operators.index_op.gremlin_example_index_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.gremlin_example_index_query.resource_path")
    def test_run(self, mock_resource_path, mock_vector_index_class):
        # Configure mocks
        mock_resource_path = "/mock/path"
        mock_vector_index_class.from_index_file.return_value = self.mock_index
        self.mock_index.search.return_value = [self.properties[0]]

        # Create a context with a query
        context = {"query": "find all persons"}

        # Create a GremlinExampleIndexQuery instance
        with patch("os.path.join", return_value=self.test_dir):
            query = GremlinExampleIndexQuery(self.embedding, num_examples=1)

            # Run the query
            result_context = query.run(context)

            # Verify the results
            self.assertIn("match_result", result_context)
            self.assertEqual(result_context["match_result"], [self.properties[0]])

            # Verify the mock was called correctly
            self.mock_index.search.assert_called_once()
            # First argument should be the embedding for "find all persons"
            args, _ = self.mock_index.search.call_args
            self.assertEqual(args[0], [1.0, 0.0, 0.0, 0.0])
            # Second argument should be num_examples (1)
            self.assertEqual(args[1], 1)

    @patch("hugegraph_llm.operators.index_op.gremlin_example_index_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.gremlin_example_index_query.resource_path")
    def test_run_with_different_query(self, mock_resource_path, mock_vector_index_class):
        # Configure mocks
        mock_resource_path = "/mock/path"
        mock_vector_index_class.from_index_file.return_value = self.mock_index
        self.mock_index.search.return_value = [self.properties[1]]

        # Create a context with a different query
        context = {"query": "count movies"}

        # Create a GremlinExampleIndexQuery instance
        with patch("os.path.join", return_value=self.test_dir):
            query = GremlinExampleIndexQuery(self.embedding, num_examples=1)

            # Run the query
            result_context = query.run(context)

            # Verify the results
            self.assertIn("match_result", result_context)
            self.assertEqual(result_context["match_result"], [self.properties[1]])

            # Verify the mock was called correctly
            self.mock_index.search.assert_called_once()
            # First argument should be the embedding for "count movies"
            args, _ = self.mock_index.search.call_args
            self.assertEqual(args[0], [0.0, 1.0, 0.0, 0.0])

    @patch("hugegraph_llm.operators.index_op.gremlin_example_index_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.gremlin_example_index_query.resource_path")
    def test_run_with_zero_examples(self, mock_resource_path, mock_vector_index_class):
        # Configure mocks
        mock_resource_path = "/mock/path"
        mock_vector_index_class.from_index_file.return_value = self.mock_index

        # Create a context with a query
        context = {"query": "find all persons"}

        # Create a GremlinExampleIndexQuery instance with num_examples=0
        with patch("os.path.join", return_value=self.test_dir):
            query = GremlinExampleIndexQuery(self.embedding, num_examples=0)

            # Run the query
            result_context = query.run(context)

            # Verify the results
            self.assertIn("match_result", result_context)
            self.assertEqual(result_context["match_result"], [])

            # Verify the mock was not called
            self.mock_index.search.assert_not_called()

    @patch("hugegraph_llm.operators.index_op.gremlin_example_index_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.gremlin_example_index_query.resource_path")
    def test_run_with_query_embedding(self, mock_resource_path, mock_vector_index_class):
        # Configure mocks
        mock_resource_path = "/mock/path"
        mock_vector_index_class.from_index_file.return_value = self.mock_index
        self.mock_index.search.return_value = [self.properties[0]]

        # Create a context with a pre-computed query embedding
        context = {"query": "find all persons", "query_embedding": [1.0, 0.0, 0.0, 0.0]}

        # Create a GremlinExampleIndexQuery instance
        with patch("os.path.join", return_value=self.test_dir):
            query = GremlinExampleIndexQuery(self.embedding, num_examples=1)

            # Run the query
            result_context = query.run(context)

            # Verify the results
            self.assertIn("match_result", result_context)
            self.assertEqual(result_context["match_result"], [self.properties[0]])

            # Verify the mock was called correctly with the pre-computed embedding
            self.mock_index.search.assert_called_once()
            args, _ = self.mock_index.search.call_args
            self.assertEqual(args[0], [1.0, 0.0, 0.0, 0.0])

    @patch("hugegraph_llm.operators.index_op.gremlin_example_index_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.gremlin_example_index_query.resource_path")
    def test_run_without_query(self, mock_resource_path, mock_vector_index_class):
        # Configure mocks
        mock_resource_path = "/mock/path"
        mock_vector_index_class.from_index_file.return_value = self.mock_index

        # Create a context without a query
        context = {}

        # Create a GremlinExampleIndexQuery instance
        with patch("os.path.join", return_value=self.test_dir):
            query = GremlinExampleIndexQuery(self.embedding, num_examples=1)

            # Run the query and expect a ValueError
            with self.assertRaises(ValueError):
                query.run(context)

    @patch("hugegraph_llm.operators.index_op.gremlin_example_index_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.gremlin_example_index_query.resource_path")
    @patch("os.path.exists")
    @patch("pandas.read_csv")
    def test_build_default_example_index(
        self, mock_read_csv, mock_exists, mock_resource_path, mock_vector_index_class
    ):
        # Configure mocks
        mock_resource_path = "/mock/path"
        mock_vector_index_class.return_value = self.mock_index
        mock_exists.return_value = False

        # Mock the CSV data
        mock_df = pd.DataFrame(self.properties)
        mock_read_csv.return_value = mock_df

        # Create a GremlinExampleIndexQuery instance
        with patch("os.path.join", return_value=self.test_dir):
            # This should trigger _build_default_example_index
            GremlinExampleIndexQuery(self.embedding, num_examples=1)

            # Verify that the index was built
            mock_vector_index_class.assert_called_once()
            self.mock_index.add.assert_called_once()
            self.mock_index.to_index_file.assert_called_once()
