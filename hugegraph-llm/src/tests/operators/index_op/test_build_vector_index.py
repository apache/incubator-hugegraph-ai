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
from hugegraph_llm.operators.index_op.build_vector_index import BuildVectorIndex


class TestBuildVectorIndex(unittest.TestCase):
    def setUp(self):
        # Create a mock embedding model
        self.mock_embedding = MagicMock(spec=BaseEmbedding)
        self.mock_embedding.get_text_embedding.return_value = [0.1, 0.2, 0.3]

        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

        # Patch the resource_path and huge_settings
        self.patcher1 = patch("hugegraph_llm.operators.index_op.build_vector_index.resource_path", self.temp_dir)
        self.patcher2 = patch("hugegraph_llm.operators.index_op.build_vector_index.huge_settings")

        self.mock_resource_path = self.patcher1.start()
        self.mock_settings = self.patcher2.start()
        self.mock_settings.graph_name = "test_graph"

        # Create the index directory
        os.makedirs(os.path.join(self.temp_dir, "test_graph", "chunks"), exist_ok=True)

        # Mock VectorIndex
        self.mock_vector_index = MagicMock(spec=VectorIndex)
        self.patcher3 = patch("hugegraph_llm.operators.index_op.build_vector_index.VectorIndex")
        self.mock_vector_index_class = self.patcher3.start()
        self.mock_vector_index_class.from_index_file.return_value = self.mock_vector_index

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

        # Stop the patchers
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()

    def test_init(self):
        # Test initialization
        builder = BuildVectorIndex(self.mock_embedding)

        # Check if the embedding is set correctly
        self.assertEqual(builder.embedding, self.mock_embedding)

        # Check if the index_dir is set correctly
        expected_index_dir = os.path.join(self.temp_dir, "test_graph", "chunks")
        self.assertEqual(builder.index_dir, expected_index_dir)

        # Check if VectorIndex.from_index_file was called with the correct path
        self.mock_vector_index_class.from_index_file.assert_called_once_with(expected_index_dir)

        # Check if the vector_index is set correctly
        self.assertEqual(builder.vector_index, self.mock_vector_index)

    def test_run_with_chunks(self):
        # Create a builder
        builder = BuildVectorIndex(self.mock_embedding)

        # Create a context with chunks
        chunks = ["chunk1", "chunk2", "chunk3"]
        context = {"chunks": chunks}

        # Run the builder
        result = builder.run(context)

        # Check if get_text_embedding was called for each chunk
        self.assertEqual(self.mock_embedding.get_text_embedding.call_count, 3)
        self.mock_embedding.get_text_embedding.assert_any_call("chunk1")
        self.mock_embedding.get_text_embedding.assert_any_call("chunk2")
        self.mock_embedding.get_text_embedding.assert_any_call("chunk3")

        # Check if add was called with the correct arguments
        expected_embeddings = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        self.mock_vector_index.add.assert_called_once_with(expected_embeddings, chunks)

        # Check if to_index_file was called with the correct path
        expected_index_dir = os.path.join(self.temp_dir, "test_graph", "chunks")
        self.mock_vector_index.to_index_file.assert_called_once_with(expected_index_dir)

        # Check if the context is returned unchanged
        self.assertEqual(result, context)

    def test_run_without_chunks(self):
        # Create a builder
        builder = BuildVectorIndex(self.mock_embedding)

        # Create a context without chunks
        context = {"other_key": "value"}

        # Run the builder and expect a ValueError
        with self.assertRaises(ValueError):
            builder.run(context)

    def test_run_with_empty_chunks(self):
        # Create a builder
        builder = BuildVectorIndex(self.mock_embedding)

        # Create a context with empty chunks
        context = {"chunks": []}

        # Run the builder
        result = builder.run(context)

        # Check if add and to_index_file were not called
        self.mock_vector_index.add.assert_not_called()
        self.mock_vector_index.to_index_file.assert_not_called()

        # Check if the context is returned unchanged
        self.assertEqual(result, context)


if __name__ == "__main__":
    unittest.main()
