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
from typing import Dict, Any, Literal

from pyhugegraph.client import PyHugeClient
from hugegraph_llm.config import resource_path, settings
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.base import BaseEmbedding


class SemanticIdQuery:
    ID_QUERY_TEMPL = "g.V().properties().hasValue('{keyword}')"
    def __init__(
            self,
            embedding: BaseEmbedding,
            by: Literal["query", "keywords"] = "keywords",
            topk_per_query: int = 10,
            topk_per_keyword: int = 1
    ):
        self.index_dir = str(os.path.join(resource_path, settings.graph_name, "graph_vids"))
        self.vector_index = VectorIndex.from_index_file(self.index_dir)
        self.embedding = embedding
        self.by = by
        self.topk_per_query = topk_per_query
        self.topk_per_keyword = topk_per_keyword
        self._client = PyHugeClient(
            settings.graph_ip,
            settings.graph_port,
            settings.graph_name,
            settings.graph_user,
            settings.graph_pwd,
            settings.graph_space,
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        graph_query_entrance = []
        if self.by == "query":
            query = context["query"]
            query_vector = self.embedding.get_text_embedding(query)
            results = self.vector_index.search(query_vector, top_k=self.topk_per_query)
            if results:
                graph_query_entrance.extend(results[:self.topk_per_query])
        else:  # by keywords
            keywords = set(context["keywords"])
            for keyword in keywords:
                resp = self._client.gremlin().exec(SemanticIdQuery.ID_QUERY_TEMPL.format(keyword=keyword))
                if len(resp['data']) > 0:
                    graph_query_entrance.append(resp['data'][0]['id'])
                else:
                    keyword_vector = self.embedding.get_text_embedding(keyword)
                    results = self.vector_index.search(keyword_vector, top_k=self.topk_per_keyword)
                    if results:
                        graph_query_entrance.extend(results[:self.topk_per_keyword])
        context["entrance_vids"] = list(set(graph_query_entrance))
        return context
