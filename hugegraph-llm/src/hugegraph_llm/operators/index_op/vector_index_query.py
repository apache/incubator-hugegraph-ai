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
from hugegraph_llm.config import resource_path, settings
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.indices.vector_index import VectorIndex


class VectorIndexQuery:
    def __init__(self, embedding: BaseEmbedding, topk: int = 3):
        self.embedding = embedding
        self.topk = topk
        index_file = str(os.path.join(resource_path, settings.graph_name, "vidx.faiss"))
        content_file = str(os.path.join(resource_path, settings.graph_name, "vidx.pkl"))
        self.vector_index = VectorIndex.from_index_file(index_file, content_file)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query")
        query_embedding = self.embedding.get_text_embedding(query)
        results = self.vector_index.search(query_embedding, self.topk)
        # TODO: check format results
        context["vector_result"] = results

        verbose = context.get("verbose") or False
        if verbose:
            print("\033[93mKNOWLEDGE FROM VECTOR:")
            print("\n".join(rel for rel in context["vector_result"]) + "\033[0m")
        return context
