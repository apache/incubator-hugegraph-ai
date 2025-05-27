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
from typing import Dict, Any, Literal, List, Tuple, Union, FrozenSet

from hugegraph_llm.config import resource_path, huge_settings
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.utils.log import log
from pyhugegraph.client import PyHugeClient


class SemanticIdQuery:
    ID_QUERY_TEMPL = "g.V({vids_str}).limit(8)"

    def __init__(
            self,
            embedding: BaseEmbedding,
            by: Literal["query", "keywords"] = "keywords",
            topk_per_query: int = 10,
            topk_per_keyword: int = huge_settings.topk_per_keyword,
            vector_dis_threshold: float = huge_settings.vector_dis_threshold,
    ):
        self.index_dir = str(os.path.join(resource_path, huge_settings.graph_name, "graph_vids"))
        self.index_dir_prop = str(os.path.join(resource_path, huge_settings.graph_name, "graph_props"))
        self.vector_index = VectorIndex.from_index_file(self.index_dir)
        self.prop_index = VectorIndex.from_index_file(self.index_dir_prop)
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
        self.schema = self._client.schema()

    def _exact_match_vids(self, keywords: List[str]) -> Tuple[List[str], List[str]]:
        assert keywords, "keywords can't be empty, please check the logic"
        # TODO: we should add a global GraphSchemaCache to avoid calling the server every time
        vertex_label_num = len(self._client.schema().getVertexLabels())
        possible_vids = set(keywords)
        for i in range(vertex_label_num):
            possible_vids.update([f"{i + 1}:{keyword}" for keyword in keywords])

        vids_str = ",".join([f"'{vid}'" for vid in possible_vids])
        resp = self._client.gremlin().exec(SemanticIdQuery.ID_QUERY_TEMPL.format(vids_str=vids_str))
        searched_vids = [v['id'] for v in resp['data']]

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
            keyword_vector = self.embedding.get_texts_embeddings([keyword])
            results = self.vector_index.search(keyword_vector[0], top_k=self.topk_per_keyword,
                                               dis_threshold=float(self.vector_dis_threshold))
            if results:
                fuzzy_match_result.extend(results[:self.topk_per_keyword])
        return fuzzy_match_result

    def _exact_match_properties(self, keywords: List[str]) -> Tuple[List[str], List[str]]:
        property_keys = self.schema.getPropertyKeys()
        log.debug("property_keys: %s", property_keys)
        matched_properties = set()
        unmatched_keywords = set(keywords)
        for key in property_keys:
            for keyword in list(unmatched_keywords):
                gremlin_query = f"g.V().has('{key.name}', '{keyword}').limit(1)"
                log.debug("prop Gremlin query: %s", gremlin_query)
                resp = self._client.gremlin().exec(gremlin_query)
                if resp.get("data"):
                    matched_properties.add((key.name, keyword))
                    unmatched_keywords.remove(keyword)
        return list(matched_properties), list(unmatched_keywords)

    def _fuzzy_match_props(self, keywords: List[str]) -> List[str]:
        fuzzy_match_result = []
        for keyword in keywords:
            keyword_vector = self.embedding.get_texts_embeddings([keyword])
            results = self.prop_index.search(keyword_vector[0], top_k=self.topk_per_keyword,
                                               dis_threshold=float(self.vector_dis_threshold))
            if results:
                fuzzy_match_result.extend(results[:self.topk_per_keyword])
        return fuzzy_match_result

    def _reformat_mixed_list_to_unique_tuples(
        self, mixed_data_list: List[Union[FrozenSet[Tuple[str, str]], Tuple[str, str]]]
    ) -> List[Tuple[str, str]]:
        unique_tuples = set()
        for item in mixed_data_list:
            if isinstance(item, (frozenset, set)):
                for prop_tuple in item:
                    if isinstance(prop_tuple, tuple) and len(prop_tuple) == 2:
                        unique_tuples.add(prop_tuple)
            elif isinstance(item, tuple):
                if len(item) == 2:
                    unique_tuples.add(item)
        return list(unique_tuples)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        graph_query_list = set()
        if self.by == "query":
            query = context["query"]
            query_vector = self.embedding.get_texts_embeddings([query])
            results = self.vector_index.search(query_vector, top_k=self.topk_per_query)
            if results:
                graph_query_list.update(results[:self.topk_per_query])
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
            index_labels = self.schema.getIndexLabels()
            if index_labels:
                props_list = set()
                exact_match_props, unmatched_props = self._exact_match_properties(keywords)
                log.debug("Exact match props: %s", exact_match_props)
                props_list.update(exact_match_props)
                fuzzy_match_props = self._fuzzy_match_props(unmatched_props)
                log.debug("Fuzzy match props: %s", fuzzy_match_props)
                props_list.update(fuzzy_match_props)
                props_list = self._reformat_mixed_list_to_unique_tuples(props_list)
                context["match_props"] = list(props_list)
                log.debug("Match props: %s", context["match_props"])
        context["match_vids"] = list(graph_query_list)
        return context
