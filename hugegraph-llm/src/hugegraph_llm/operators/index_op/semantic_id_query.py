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
from typing import Any, Dict, List, Literal, Tuple

<<<<<<< HEAD
<<<<<<< HEAD
from hugegraph_llm.config import resource_path, huge_settings, llm_settings
from hugegraph_llm.indices.vector_index import VectorIndex
=======
from hugegraph_llm.config import resource_path, huge_settings, llm_settings
from hugegraph_llm.indices.vector_index.faiss_vector_store import FaissVectorIndex
>>>>>>> 902fee5 (feat(llm): some type bug && revert to FaissVectorIndex)
=======
from pyhugegraph.client import PyHugeClient

from hugegraph_llm.config import huge_settings, resource_path
from hugegraph_llm.indices.vector_index.base import VectorStoreBase
>>>>>>> 38dce0b (feat(llm): vector db finished)
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.utils.embedding_utils import get_filename_prefix, get_index_folder_name
from hugegraph_llm.utils.log import log


class SemanticIdQuery:
    ID_QUERY_TEMPL = "g.V({vids_str}).limit(8)"

    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_index: type[VectorStoreBase],
        by: Literal["query", "keywords"] = "keywords",
        topk_per_query: int = 10,
        topk_per_keyword: int = huge_settings.topk_per_keyword,
        vector_dis_threshold: float = huge_settings.vector_dis_threshold,
    ):
<<<<<<< HEAD
<<<<<<< HEAD
        self.folder_name = get_index_folder_name(
            huge_settings.graph_name, huge_settings.graph_space
        )
        self.index_dir = str(os.path.join(resource_path, self.folder_name, "graph_vids"))
        self.filename_prefix = get_filename_prefix(
            llm_settings.embedding_type, getattr(embedding, "model_name", None)
        )
        self.vector_index = VectorIndex.from_index_file(self.index_dir, self.filename_prefix)
=======
        self.folder_name = get_index_folder_name(
            huge_settings.graph_name, huge_settings.graph_space
        )
        self.index_dir = str(os.path.join(resource_path, self.folder_name, "graph_vids"))
        self.filename_prefix = get_filename_prefix(
            llm_settings.embedding_type, getattr(embedding, "model_name", None)
        )
        self.vector_index = FaissVectorIndex.from_index_file(self.index_dir, self.filename_prefix)
>>>>>>> 902fee5 (feat(llm): some type bug && revert to FaissVectorIndex)
=======
        self.index_dir = str(os.path.join(resource_path, huge_settings.graph_name, "graph_vids"))
        self.vector_index = vector_index.from_name(
            embedding.get_embedding_dim(), huge_settings.graph_name, "graph_vids"
        )
>>>>>>> 38dce0b (feat(llm): vector db finished)
        self.embedding = embedding
        self.by = by
        self.topk_per_query = topk_per_query
        self.topk_per_keyword = topk_per_keyword
        self.vector_dis_threshold = vector_dis_threshold
        self._client = PyHugeClient(
            url=huge_settings.graph_url,
            graph=huge_settings.graph_name,
            user=huge_settings.graph_user,
            pwd=huge_settings.graph_pwd,
            graphspace=huge_settings.graph_space,
        )

    def _exact_match_vids(self, keywords: List[str]) -> Tuple[List[str], List[str]]:
        assert keywords, "keywords can't be empty, please check the logic"
        # TODO: we should add a global GraphSchemaCache to avoid calling the server every time
        vertex_label_num = len(self._client.schema().getVertexLabels())
        possible_vids = set(keywords)
        for i in range(vertex_label_num):
            possible_vids.update([f"{i + 1}:{keyword}" for keyword in keywords])

        vids_str = ",".join([f"'{vid}'" for vid in possible_vids])
        resp = self._client.gremlin().exec(SemanticIdQuery.ID_QUERY_TEMPL.format(vids_str=vids_str))
        searched_vids = [v["id"] for v in resp["data"]]

        unsearched_keywords = set(keywords)
        for vid in searched_vids:
            for keyword in unsearched_keywords:
                if keyword in vid:
                    unsearched_keywords.remove(keyword)
                    break
        return searched_vids, list(unsearched_keywords)

    def _fuzzy_match_vids(self, keywords: List[str]) -> List[str]:
        fuzzy_match_result = []
        for keyword in keywords:
<<<<<<< HEAD
            keyword_vector = self.embedding.get_texts_embeddings([keyword])[0]
            results = self.vector_index.search(
                keyword_vector,
                top_k=self.topk_per_keyword,
                dis_threshold=float(self.vector_dis_threshold),
=======
            keyword_vector = self.embedding.get_texts_embeddings([keyword])[0]
            results = self.vector_index.search(
                keyword_vector, top_k=self.topk_per_keyword, dis_threshold=float(self.vector_dis_threshold)
>>>>>>> 902fee5 (feat(llm): some type bug && revert to FaissVectorIndex)
            )
            if results:
                fuzzy_match_result.extend(results[: self.topk_per_keyword])
        return fuzzy_match_result

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        graph_query_list = set()
        if self.by == "query":
            query = context["query"]
            query_vector = self.embedding.get_texts_embeddings([query])[0]
            results = self.vector_index.search(query_vector, top_k=self.topk_per_query)
            if results:
                graph_query_list.update(results[: self.topk_per_query])
        else:  # by keywords
            keywords = context.get("keywords", [])
            if not keywords:
                context["match_vids"] = []
                return context

            exact_match_vids, unmatched_vids = self._exact_match_vids(keywords)
            graph_query_list.update(exact_match_vids)
            fuzzy_match_vids = self._fuzzy_match_vids(unmatched_vids)
            log.debug("Fuzzy match vids: %s", fuzzy_match_vids)
            graph_query_list.update(fuzzy_match_vids)
        context["match_vids"] = list(graph_query_list)
        return context
