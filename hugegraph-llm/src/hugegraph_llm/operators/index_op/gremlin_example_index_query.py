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
from typing import Dict, Any

from hugegraph_llm.config import resource_path
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.indices.vector_index import VectorIndex


class GremlinExampleIndexQuery:
    def __init__(self, query: str, embedding: BaseEmbedding, num_examples: int = 1):
        self.query = query
        self.embedding = embedding
        self.num_examples = num_examples
        self.index_dir = os.path.join(resource_path, "gremlin_examples")
        self.vector_index = VectorIndex.from_index_file(self.index_dir)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["query"] = self.query
        query_embedding = self.embedding.get_text_embedding(self.query)
        context["match_result"] = self.vector_index.search(query_embedding, self.num_examples, dis_threshold=2)
        return context
