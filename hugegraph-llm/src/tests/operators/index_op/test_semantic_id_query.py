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
from hugegraph_llm.operators.index_op.semantic_id_query import SemanticIdQuery


class MockEmbedding(BaseEmbedding):
    """Mock embedding class for testing"""

    def __init__(self):
        self.model = "mock_model"

    def get_text_embedding(self, text):
        # Return a simple mock embedding based on the text
        if text == "query1":
            return [1.0, 0.0, 0.0, 0.0]
        if text == "keyword1":
            return [0.0, 1.0, 0.0, 0.0]
        if text == "keyword2":
            return [0.0, 0.0, 1.0, 0.0]
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


class MockPyHugeClient:
    """Mock PyHugeClient for testing"""

    def __init__(self, *args, **kwargs):
        self._schema = MagicMock()
        self._schema.getVertexLabels.return_value = ["person", "movie"]
        self._gremlin = MagicMock()
        self._gremlin.exec.return_value = {
            "data": [
                {"id": "1:keyword1", "properties": {"name": "keyword1"}},
                {"id": "2:keyword2", "properties": {"name": "keyword2"}},
            ]
        }

    def schema(self):
        return self._schema

    def gremlin(self):
        return self._gremlin


class TestSemanticIdQuery(unittest.TestCase):
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
        self.properties = ["1:vid1", "2:vid2", "3:vid3", "4:vid4"]

        # Create a mock vector index
        self.mock_index = MagicMock()
        self.mock_index.search.return_value = ["1:vid1"]  # Default return value

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    @patch("hugegraph_llm.operators.index_op.semantic_id_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.semantic_id_query.resource_path")
    @patch("hugegraph_llm.operators.index_op.semantic_id_query.huge_settings")
    @patch("hugegraph_llm.operators.index_op.semantic_id_query.PyHugeClient", new=MockPyHugeClient)
    def test_init(self, mock_settings, mock_resource_path, mock_vector_index_class):
        # Configure mocks
        mock_settings.graph_name = "test_graph"
        mock_settings.topk_per_keyword = 5
        mock_resource_path = "/mock/path"
        mock_vector_index_class.from_index_file.return_value = self.mock_index

        # Create a SemanticIdQuery instance
        with patch("os.path.join", return_value=self.test_dir):
            query = SemanticIdQuery(self.embedding, by="query", topk_per_query=3)

            # Verify the instance was initialized correctly
            self.assertEqual(query.embedding, self.embedding)
            self.assertEqual(query.by, "query")
            self.assertEqual(query.topk_per_query, 3)
            self.assertEqual(query.vector_index, self.mock_index)
            mock_vector_index_class.from_index_file.assert_called_once()

    @patch("hugegraph_llm.operators.index_op.semantic_id_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.semantic_id_query.resource_path")
    @patch("hugegraph_llm.operators.index_op.semantic_id_query.huge_settings")
    @patch("hugegraph_llm.operators.index_op.semantic_id_query.PyHugeClient", new=MockPyHugeClient)
    def test_run_by_query(self, mock_settings, mock_resource_path, mock_vector_index_class):
        # Configure mocks
        mock_settings.graph_name = "test_graph"
        mock_settings.topk_per_keyword = 5
        mock_resource_path = "/mock/path"
        mock_vector_index_class.from_index_file.return_value = self.mock_index
        self.mock_index.search.return_value = ["1:vid1", "2:vid2"]

        # Create a context with a query
        context = {"query": "query1"}

        # Create a SemanticIdQuery instance
        with patch("os.path.join", return_value=self.test_dir):
            query = SemanticIdQuery(self.embedding, by="query", topk_per_query=2)

            # Run the query
            result_context = query.run(context)

            # Verify the results
            self.assertIn("match_vids", result_context)
            self.assertEqual(set(result_context["match_vids"]), {"1:vid1", "2:vid2"})

            # Verify the mock was called correctly
            self.mock_index.search.assert_called_once()
            # First argument should be the embedding for "query1"
            args, kwargs = self.mock_index.search.call_args
            self.assertEqual(args[0], [1.0, 0.0, 0.0, 0.0])
            self.assertEqual(kwargs.get("top_k"), 2)

    @patch("hugegraph_llm.operators.index_op.semantic_id_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.semantic_id_query.resource_path")
    @patch("hugegraph_llm.operators.index_op.semantic_id_query.huge_settings")
    @patch("hugegraph_llm.operators.index_op.semantic_id_query.PyHugeClient", new=MockPyHugeClient)
    def test_run_by_keywords(self, mock_settings, mock_resource_path, mock_vector_index_class):
        # Configure mocks
        mock_settings.graph_name = "test_graph"
        mock_settings.topk_per_keyword = 2
        mock_settings.vector_dis_threshold = 1.5
        mock_resource_path = "/mock/path"
        mock_vector_index_class.from_index_file.return_value = self.mock_index
        self.mock_index.search.return_value = ["3:vid3", "4:vid4"]

        # Create a context with keywords
        # Use a keyword that won't be found by exact match to ensure fuzzy matching is used
        context = {"keywords": ["unknown_keyword", "another_unknown"]}

        # Mock the _exact_match_vids method to return empty results for these keywords
        with patch.object(MockPyHugeClient, "gremlin") as mock_gremlin:
            mock_gremlin.return_value.exec.return_value = {"data": []}

            # Create a SemanticIdQuery instance
            with patch("os.path.join", return_value=self.test_dir):
                query = SemanticIdQuery(self.embedding, by="keywords", topk_per_keyword=2)

                # Run the query
                result_context = query.run(context)

                # Verify the results
                self.assertIn("match_vids", result_context)
                # Should include fuzzy matches from the index
                self.assertEqual(set(result_context["match_vids"]), {"3:vid3", "4:vid4"})

                # Verify the mock was called correctly for fuzzy matching
                self.mock_index.search.assert_called()

    @patch("hugegraph_llm.operators.index_op.semantic_id_query.VectorIndex")
    @patch("hugegraph_llm.operators.index_op.semantic_id_query.resource_path")
    @patch("hugegraph_llm.operators.index_op.semantic_id_query.huge_settings")
    @patch("hugegraph_llm.operators.index_op.semantic_id_query.PyHugeClient", new=MockPyHugeClient)
    def test_run_with_empty_keywords(
        self, mock_settings, mock_resource_path, mock_vector_index_class
    ):
        # Configure mocks
        mock_settings.graph_name = "test_graph"
        mock_settings.topk_per_keyword = 5
        mock_resource_path = "/mock/path"
        mock_vector_index_class.from_index_file.return_value = self.mock_index

        # Create a context with empty keywords
        context = {"keywords": []}

        # Create a SemanticIdQuery instance
        with patch("os.path.join", return_value=self.test_dir):
            query = SemanticIdQuery(self.embedding, by="keywords")

            # Run the query
            result_context = query.run(context)

            # Verify the results
            self.assertIn("match_vids", result_context)
            self.assertEqual(result_context["match_vids"], [])

            # Verify the mock was not called
            self.mock_index.search.assert_not_called()
