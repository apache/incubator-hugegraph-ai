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
import unittest

from hugegraph_llm.models.embeddings.base import SimilarityMode
from hugegraph_llm.models.embeddings.ollama import OllamaEmbedding


class TestOllamaEmbedding(unittest.TestCase):
    def setUp(self):
        self.skip_external = os.getenv("SKIP_EXTERNAL_SERVICES", "false").lower() == "true"

    @unittest.skipIf(os.getenv("SKIP_EXTERNAL_SERVICES", "false").lower() == "true",
                     "Skipping external service tests")
    def test_get_text_embedding(self):
        ollama_embedding = OllamaEmbedding(model_name="quentinz/bge-large-zh-v1.5")
        embedding = ollama_embedding.get_text_embedding("hello world")
        print(embedding)

    @unittest.skipIf(os.getenv("SKIP_EXTERNAL_SERVICES", "false").lower() == "true",
                     "Skipping external service tests")
    def test_get_cosine_similarity(self):
        ollama_embedding = OllamaEmbedding(model_name="quentinz/bge-large-zh-v1.5")
        embedding1 = ollama_embedding.get_text_embedding("hello world")
        embedding2 = ollama_embedding.get_text_embedding("bye world")
        similarity = OllamaEmbedding.similarity(embedding1, embedding2, SimilarityMode.DEFAULT)
        print(similarity)
