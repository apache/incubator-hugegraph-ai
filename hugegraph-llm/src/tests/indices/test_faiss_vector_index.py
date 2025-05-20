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
from pprint import pprint

from hugegraph_llm.indices.vector_index.faiss_vector_store import FaissVectorIndex
from hugegraph_llm.models.embeddings.ollama import OllamaEmbedding


class TestVectorIndex(unittest.TestCase):
    def test_vector_index(self):
        embedder = OllamaEmbedding("quentinz/bge-large-zh-v1.5")
        data = [
            "腾讯的合伙人有字节跳动",
            "谷歌和微软是竞争关系",
            "美团的合伙人有字节跳动",
        ]
        data_embedding = [embedder.get_text_embedding(d) for d in data]
        index = FaissVectorIndex(1024)
        index.add(data_embedding, data)
        query = "腾讯的合伙人有哪些？"
        query_vector = embedder.get_text_embedding(query)
        results = index.search(query_vector, 2)
        pprint(results)
