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

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.operators.index_op.build_gremlin_example_index import BuildGremlinExampleIndex


class TestBuildGremlinExampleIndex(unittest.TestCase):
    def setUp(self):
        # Create a mock embedding model
        self.mock_embedding = MagicMock(spec=BaseEmbedding)
        self.mock_embedding.get_text_embedding.return_value = [0.1, 0.2, 0.3]

        # Create example data
        self.examples = [
            {"query": "g.V().hasLabel('person')", "description": "Find all persons"},
            {"query": "g.V().hasLabel('movie')", "description": "Find all movies"},
        ]

        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

        # Patch the resource_path
        self.patcher1 = patch(
            "hugegraph_llm.operators.index_op.build_gremlin_example_index.resource_path", self.temp_dir
        )
        self.patcher1.start()

        # Mock the new utility functions
        self.patcher2 = patch("hugegraph_llm.operators.index_op.build_gremlin_example_index.get_index_folder_name")
        self.mock_get_index_folder_name = self.patcher2.start()
        self.mock_get_index_folder_name.return_value = "hugegraph"

        self.patcher3 = patch("hugegraph_llm.operators.index_op.build_gremlin_example_index.get_filename_prefix")
        self.mock_get_filename_prefix = self.patcher3.start()
        self.mock_get_filename_prefix.return_value = "test_prefix"

        self.patcher4 = patch("hugegraph_llm.operators.index_op.build_gremlin_example_index.get_embeddings_parallel")
        self.mock_get_embeddings_parallel = self.patcher4.start()
        self.mock_get_embeddings_parallel.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        # Mock VectorIndex
        self.mock_vector_index = MagicMock(spec=VectorIndex)
        self.patcher5 = patch("hugegraph_llm.operators.index_op.build_gremlin_example_index.VectorIndex")
        self.mock_vector_index_class = self.patcher5.start()
        self.mock_vector_index_class.return_value = self.mock_vector_index

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

        # Stop the patchers
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()

    def test_init(self):
        # Test initialization
        builder = BuildGremlinExampleIndex(self.mock_embedding, self.examples)

        # Check if the embedding is set correctly
        self.assertEqual(builder.embedding, self.mock_embedding)

        # Check if the examples are set correctly
        self.assertEqual(builder.examples, self.examples)

        # Check if the index_dir is set correctly (now includes folder structure)
        expected_index_dir = os.path.join(self.temp_dir, "hugegraph", "gremlin_examples")
        self.assertEqual(builder.index_dir, expected_index_dir)

    def test_run_with_examples(self):
        # Create a builder
        builder = BuildGremlinExampleIndex(self.mock_embedding, self.examples)

        # Create a context
        context = {}

        # Run the builder
        result = builder.run(context)

        # Check if get_embeddings_parallel was called
        self.mock_get_embeddings_parallel.assert_called_once()

        # Check if VectorIndex was initialized with the correct dimension
        self.mock_vector_index_class.assert_called_once_with(3)  # dimension of [0.1, 0.2, 0.3]

        # Check if add was called with the correct arguments
        expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # from mock return value
        self.mock_vector_index.add.assert_called_once_with(expected_embeddings, self.examples)

        # Check if to_index_file was called with the correct path and prefix
        expected_index_dir = os.path.join(self.temp_dir, "hugegraph", "gremlin_examples")
        self.mock_vector_index.to_index_file.assert_called_once_with(expected_index_dir, "test_prefix")

        # Check if the context is updated correctly
        expected_context = {"embed_dim": 3}
        self.assertEqual(result, expected_context)

    def test_run_with_empty_examples(self):
        # Create a builder with empty examples
        builder = BuildGremlinExampleIndex(self.mock_embedding, [])

        # Create a context
        context = {"test": "value"}

        # The run method should handle empty examples gracefully
        result = builder.run(context)

        # Should return embed_dim as 0 for empty examples
        self.assertEqual(result["embed_dim"], 0)
        self.assertEqual(result["test"], "value")  # Original context should be preserved

        # Check if VectorIndex was not initialized
        self.mock_vector_index_class.assert_not_called()

        # Check if add and to_index_file were not called
        self.mock_vector_index.add.assert_not_called()
        self.mock_vector_index.to_index_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
