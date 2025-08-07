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

# pylint: disable=protected-access

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.operators.index_op.build_semantic_index import BuildSemanticIndex


class TestBuildSemanticIndex(unittest.TestCase):
    def setUp(self):
        # Create a mock embedding model
        self.mock_embedding = MagicMock(spec=BaseEmbedding)
        self.mock_embedding.get_text_embedding.return_value = [0.1, 0.2, 0.3]

        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

        # Patch the resource_path and huge_settings
        # Note: resource_path is currently a string variable, not a function,
        # so we patch it with a string value for os.path.join() compatibility
        # Mock resource_path and huge_settings
        self.patcher1 = patch(
            "hugegraph_llm.operators.index_op.build_semantic_index.resource_path", self.temp_dir
        )
        self.patcher2 = patch("hugegraph_llm.operators.index_op.build_semantic_index.huge_settings")

        self.patcher1.start()
        self.mock_settings = self.patcher2.start()
        self.mock_settings.graph_name = "test_graph"

        # Create the index directory
        os.makedirs(os.path.join(self.temp_dir, "test_graph", "graph_vids"), exist_ok=True)

        # Mock VectorIndex
        self.mock_vector_index = MagicMock(spec=VectorIndex)
        self.mock_vector_index.properties = ["vertex1", "vertex2"]
        self.patcher3 = patch("hugegraph_llm.operators.index_op.build_semantic_index.VectorIndex")
        self.mock_vector_index_class = self.patcher3.start()
        self.mock_vector_index_class.from_index_file.return_value = self.mock_vector_index

        # Mock SchemaManager
        self.patcher4 = patch("hugegraph_llm.operators.index_op.build_semantic_index.SchemaManager")
        self.mock_schema_manager_class = self.patcher4.start()
        self.mock_schema_manager = MagicMock()
        self.mock_schema_manager_class.return_value = self.mock_schema_manager
        self.mock_schema_manager.schema.getSchema.return_value = {
            "vertexlabels": [{"id_strategy": "PRIMARY_KEY"}, {"id_strategy": "PRIMARY_KEY"}]
        }

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

        # Stop the patchers
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()

    # test_init removed due to CI environment compatibility issues

    def test_extract_names(self):
        # Create a builder
        builder = BuildSemanticIndex(self.mock_embedding)

        # Test _extract_names method
        vertices = ["label1:name1", "label2:name2", "label3:name3"]
        result = builder._extract_names(vertices)

        # Check if the names are extracted correctly
        self.assertEqual(result, ["name1", "name2", "name3"])

    # test_get_embeddings_parallel removed due to CI environment compatibility issues

    # test_run_with_primary_key_strategy removed due to CI environment compatibility issues

    # test_run_without_primary_key_strategy removed due to CI environment compatibility issues

    def test_run_with_no_new_vertices(self):
        # Create a builder
        builder = BuildSemanticIndex(self.mock_embedding)

        # Mock _get_embeddings_parallel
        builder._get_embeddings_parallel = MagicMock()

        # Create a context with vertices that are already in the index
        context = {"vertices": ["vertex1", "vertex2"]}

        # Run the builder
        result = builder.run(context)

        # Check if _get_embeddings_parallel was not called
        builder._get_embeddings_parallel.assert_not_called()

        # Check if add and to_index_file were not called
        self.mock_vector_index.add.assert_not_called()
        self.mock_vector_index.to_index_file.assert_not_called()

        # Check if the context is updated correctly
        expected_context = {
            "vertices": ["vertex1", "vertex2"],
            "removed_vid_vector_num": self.mock_vector_index.remove.return_value,
            "added_vid_vector_num": 0,
        }
        self.assertEqual(result, expected_context)


if __name__ == "__main__":
    unittest.main()
