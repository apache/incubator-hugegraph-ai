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
from typing import Dict, Any, List

from hugegraph_llm.config import resource_path
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.base import BaseEmbedding


# FIXME: we need keep the logic same with build_semantic_index.py
class BuildGremlinExampleIndex:
    def __init__(self, embedding: BaseEmbedding, examples: List[Dict[str, str]]):
        self.index_dir = os.path.join(resource_path, "gremlin_examples")
        self.examples = examples
        self.embedding = embedding

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        examples_embedding = []
        for example in self.examples:
            examples_embedding.append(self.embedding.get_text_embedding(example["query"]))
        embed_dim = len(examples_embedding[0])
        if len(self.examples) > 0:
            vector_index = VectorIndex(embed_dim)
            vector_index.add(examples_embedding, self.examples)
            vector_index.to_index_file(self.index_dir)
        context["embed_dim"] = embed_dim
        return context
