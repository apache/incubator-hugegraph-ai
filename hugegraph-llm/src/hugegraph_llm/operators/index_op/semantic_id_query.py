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

from hugegraph_llm.config import resource_path, settings
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.base import BaseEmbedding


class SemanticIdQuery:
    def __init__(self, embedding: BaseEmbedding):
        index_file = str(os.path.join(resource_path, settings.graph_name, "vid.faiss"))
        content_file = str(os.path.join(resource_path, settings.graph_name, "vid.pkl"))
        self.vector_index = VectorIndex.from_index_file(index_file, content_file)
        self.embedding = embedding

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        keywords = context["keywords"]
        graph_query_entrance = []
        for keyword in keywords:
            query_vector = self.embedding.get_text_embedding(keyword)
            results = self.vector_index.search(query_vector, top_k=1)
            if results:
                graph_query_entrance.append(results[0])
        context["entrance_vids"] = list(set(graph_query_entrance))
        return context
