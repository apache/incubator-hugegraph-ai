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
from unittest.mock import MagicMock, patch, mock_open, ANY, call
import os
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor

from hugegraph_llm.operators.index_op.build_semantic_index import BuildSemanticIndex
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.indices.vector_index import VectorIndex


class TestBuildSemanticIndex(unittest.TestCase):
    def setUp(self):
        # Create a mock embedding model
        self.mock_embedding = MagicMock(spec=BaseEmbedding)
        self.mock_embedding.get_text_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Patch the resource_path and huge_settings
        self.patcher1 = patch('hugegraph_llm.operators.index_op.build_semantic_index.resource_path', self.temp_dir)
        self.patcher2 = patch('hugegraph_llm.operators.index_op.build_semantic_index.huge_settings')
        
        self.mock_resource_path = self.patcher1.start()
        self.mock_settings = self.patcher2.start()
        self.mock_settings.graph_name = "test_graph"
        
        # Create the index directory
        os.makedirs(os.path.join(self.temp_dir, "test_graph", "graph_vids"), exist_ok=True)
        
        # Mock VectorIndex
        self.mock_vector_index = MagicMock(spec=VectorIndex)
        self.mock_vector_index.properties = ["vertex1", "vertex2"]
        self.patcher3 = patch('hugegraph_llm.operators.index_op.build_semantic_index.VectorIndex')
        self.mock_vector_index_class = self.patcher3.start()
        self.mock_vector_index_class.from_index_file.return_value = self.mock_vector_index
        
        # Mock SchemaManager
        self.patcher4 = patch('hugegraph_llm.operators.index_op.build_semantic_index.SchemaManager')
        self.mock_schema_manager_class = self.patcher4.start()
        self.mock_schema_manager = MagicMock()
        self.mock_schema_manager_class.return_value = self.mock_schema_manager
        self.mock_schema_manager.schema.getSchema.return_value = {
            "vertexlabels": [
                {"id_strategy": "PRIMARY_KEY"},
                {"id_strategy": "PRIMARY_KEY"}
            ]
        }

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Stop the patchers
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()

    def test_init(self):
        # Test initialization
        builder = BuildSemanticIndex(self.mock_embedding)
        
        # Check if the embedding is set correctly
        self.assertEqual(builder.embedding, self.mock_embedding)
        
        # Check if the index_dir is set correctly
        expected_index_dir = os.path.join(self.temp_dir, "test_graph", "graph_vids")
        self.assertEqual(builder.index_dir, expected_index_dir)
        
        # Check if VectorIndex.from_index_file was called with the correct path
        self.mock_vector_index_class.from_index_file.assert_called_once_with(expected_index_dir)
        
        # Check if the vid_index is set correctly
        self.assertEqual(builder.vid_index, self.mock_vector_index)
        
        # Check if SchemaManager was initialized with the correct graph name
        self.mock_schema_manager_class.assert_called_once_with("test_graph")
        
        # Check if the schema manager is set correctly
        self.assertEqual(builder.sm, self.mock_schema_manager)

    def test_extract_names(self):
        # Create a builder
        builder = BuildSemanticIndex(self.mock_embedding)
        
        # Test _extract_names method
        vertices = ["label1:name1", "label2:name2", "label3:name3"]
        result = builder._extract_names(vertices)
        
        # Check if the names are extracted correctly
        self.assertEqual(result, ["name1", "name2", "name3"])

    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_get_embeddings_parallel(self, mock_executor_class):
        # Create a builder
        builder = BuildSemanticIndex(self.mock_embedding)
        
        # Setup mock executor
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        mock_executor.map.return_value = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        
        # Test _get_embeddings_parallel method
        vids = ["vid1", "vid2", "vid3"]
        result = builder._get_embeddings_parallel(vids)
        
        # Check if ThreadPoolExecutor.map was called with the correct arguments
        mock_executor.map.assert_called_once_with(self.mock_embedding.get_text_embedding, vids)
        
        # Check if the result is correct
        self.assertEqual(result, [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])

    def test_run_with_primary_key_strategy(self):
        # Create a builder
        builder = BuildSemanticIndex(self.mock_embedding)
        
        # Mock _get_embeddings_parallel
        builder._get_embeddings_parallel = MagicMock()
        builder._get_embeddings_parallel.return_value = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        
        # Create a context with vertices that have proper format for PRIMARY_KEY strategy
        context = {"vertices": ["label1:name1", "label2:name2", "label3:name3"]}
        
        # Run the builder
        result = builder.run(context)
        
        # We can't directly assert what was passed to remove since it's a set and order is not guaranteed
        # Instead, we'll check that remove was called once and then verify the result context
        self.mock_vector_index.remove.assert_called_once()
        removed_set = self.mock_vector_index.remove.call_args[0][0]
        self.assertIsInstance(removed_set, set)
        # The set should contain vertex1 and vertex2 (the past_vids) that are not in present_vids
        self.assertIn("vertex1", removed_set)
        self.assertIn("vertex2", removed_set)
        
        # Check if _get_embeddings_parallel was called with the correct arguments
        # Since all vertices have PRIMARY_KEY strategy, we should extract names
        builder._get_embeddings_parallel.assert_called_once()
        # Get the actual arguments passed to _get_embeddings_parallel
        args = builder._get_embeddings_parallel.call_args[0][0]
        # Check that the arguments contain the expected names
        self.assertEqual(set(args), set(["name1", "name2", "name3"]))
        
        # Check if add was called with the correct arguments
        self.mock_vector_index.add.assert_called_once()
        # Get the actual arguments passed to add
        add_args = self.mock_vector_index.add.call_args
        # Check that the embeddings and vertices are correct
        self.assertEqual(add_args[0][0], [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
        self.assertEqual(set(add_args[0][1]), set(["label1:name1", "label2:name2", "label3:name3"]))
        
        # Check if to_index_file was called with the correct path
        expected_index_dir = os.path.join(self.temp_dir, "test_graph", "graph_vids")
        self.mock_vector_index.to_index_file.assert_called_once_with(expected_index_dir)
        
        # Check if the context is updated correctly
        self.assertEqual(result["vertices"], ["label1:name1", "label2:name2", "label3:name3"])
        self.assertEqual(result["removed_vid_vector_num"], self.mock_vector_index.remove.return_value)
        self.assertEqual(result["added_vid_vector_num"], 3)

    def test_run_without_primary_key_strategy(self):
        # Create a builder
        builder = BuildSemanticIndex(self.mock_embedding)
        
        # Change the schema to not use PRIMARY_KEY strategy
        self.mock_schema_manager.schema.getSchema.return_value = {
            "vertexlabels": [
                {"id_strategy": "AUTOMATIC"},
                {"id_strategy": "AUTOMATIC"}
            ]
        }
        
        # Mock _get_embeddings_parallel
        builder._get_embeddings_parallel = MagicMock()
        builder._get_embeddings_parallel.return_value = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        
        # Create a context with vertices
        context = {"vertices": ["label1:name1", "label2:name2", "label3:name3"]}
        
        # Run the builder
        result = builder.run(context)
        
        # Check if _get_embeddings_parallel was called with the correct arguments
        # Since vertices don't have PRIMARY_KEY strategy, we should use the original vertex IDs
        builder._get_embeddings_parallel.assert_called_once()
        # Get the actual arguments passed to _get_embeddings_parallel
        args = builder._get_embeddings_parallel.call_args[0][0]
        # Check that the arguments contain the expected vertex IDs
        self.assertEqual(set(args), set(["label1:name1", "label2:name2", "label3:name3"]))
        
        # Check if the context is updated correctly
        self.assertEqual(result["vertices"], ["label1:name1", "label2:name2", "label3:name3"])
        self.assertEqual(result["removed_vid_vector_num"], self.mock_vector_index.remove.return_value)
        self.assertEqual(result["added_vid_vector_num"], 3)

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
            "added_vid_vector_num": 0
        }
        self.assertEqual(result, expected_context)


if __name__ == "__main__":
    unittest.main() 