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

from hugegraph_llm.indices.vector_index.milvus_vector_store import MilvusVectorIndex
from hugegraph_llm.models.embeddings.ollama import OllamaEmbedding

test_name = "test"


class TestMilvusVectorIndex(unittest.TestCase):
    def tearDown(self):
        MilvusVectorIndex.clean(test_name)

    def test_vector_index(self):
        embedder = OllamaEmbedding("quentinz/bge-large-zh-v1.5")

        data = [
            "腾讯的合伙人有字节跳动",
            "谷歌和微软是竞争关系",
            "美团的合伙人有字节跳动",
        ]
        data_embedding = [embedder.get_text_embedding(d) for d in data]

        index = MilvusVectorIndex.from_name(1024, test_name)
        index.add(data_embedding, data)

        query = "腾讯的合伙人有哪些？"
        query_vector = embedder.get_text_embedding(query)
        results = index.search(query_vector, 2, dis_threshold=1000)
        pprint(results)

        self.assertIsNotNone(results)
        self.assertLessEqual(len(results), 2)

    def test_save_and_load(self):
        embedder = OllamaEmbedding("quentinz/bge-large-zh-v1.5")

        data = [
            "腾讯的合伙人有字节跳动",
            "谷歌和微软是竞争关系",
            "美团的合伙人有字节跳动",
        ]
        data_embedding = [embedder.get_text_embedding(d) for d in data]

        index = MilvusVectorIndex.from_name(1024, test_name)
        index.add(data_embedding, data)

        index.save_index_by_name(test_name)

        loaded_index = MilvusVectorIndex.from_name(1024, test_name)

        query = "腾讯的合伙人有哪些？"
        query_vector = embedder.get_text_embedding(query)
        results = loaded_index.search(query_vector, 2, dis_threshold=1000)

        self.assertIsNotNone(results)
        self.assertLessEqual(len(results), 2)

    def test_remove_entries(self):
        embedder = OllamaEmbedding("quentinz/bge-large-zh-v1.5")
        data = [
            "腾讯的合伙人有字节跳动",
            "谷歌和微软是竞争关系",
            "美团的合伙人有字节跳动",
        ]
        data_embedding = [embedder.get_text_embedding(d) for d in data]

        index = MilvusVectorIndex.from_name(1024, test_name)
        index.add(data_embedding, data)

        query = "合伙人"
        query_vector = embedder.get_text_embedding(query)
        initial_results = index.search(query_vector, 3, dis_threshold=1000)
        initial_count = len(initial_results)

        remove_count = index.remove(["谷歌和微软是竞争关系"])

        self.assertEqual(remove_count, 1)

        after_results = index.search(query_vector, 3, dis_threshold=1000)
        self.assertLessEqual(len(after_results), initial_count - 1)
        self.assertNotIn("谷歌和微软是竞争关系", after_results)
