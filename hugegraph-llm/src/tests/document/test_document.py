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
import importlib


class TestDocumentModule(unittest.TestCase):
    def test_import_document_module(self):
        """Test that the document module can be imported."""
        try:
            import hugegraph_llm.document
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import hugegraph_llm.document module")
            
    def test_import_chunk_split(self):
        """Test that the chunk_split module can be imported."""
        try:
            from hugegraph_llm.document import chunk_split
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import chunk_split module")
            
    def test_chunk_splitter_class_exists(self):
        """Test that the ChunkSplitter class exists in the chunk_split module."""
        try:
            from hugegraph_llm.document.chunk_split import ChunkSplitter
            self.assertTrue(True)
        except ImportError:
            self.fail("ChunkSplitter class not found in chunk_split module")
            
    def test_module_reload(self):
        """Test that the document module can be reloaded."""
        try:
            import hugegraph_llm.document
            importlib.reload(hugegraph_llm.document)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Failed to reload document module: {e}")
