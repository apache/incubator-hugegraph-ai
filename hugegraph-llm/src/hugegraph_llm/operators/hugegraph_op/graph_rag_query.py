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

import json
import os
import re
from typing import Any, Dict, Optional, List, Set, Tuple

from hugegraph_llm.config import settings, resource_path
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.utils.log import log
from pyhugegraph.client import PyHugeClient

# TODO: remove 'as('subj)' step
VERTEX_QUERY_TPL = "g.V({keywords}).limit(8).as('subj').toList()"

# TODO: we could use a simpler query (like kneighbor-api to get the edges)
# TODO: test with profile()/explain() to speed up the query
VID_QUERY_NEIGHBOR_TPL = """\
g.V({keywords})
.repeat(
   bothE({edge_labels}).limit({edge_limit}).otherV().dedup()
).times({max_deep}).emit()
.simplePath()
.path()
.by(project('label', 'id', 'props')
   .by(label())
   .by(id())
   .by(valueMap().by(unfold()))
)
.by(project('label', 'inV', 'outV', 'props')
   .by(label())
   .by(inV().id())
   .by(outV().id())
   .by(valueMap().by(unfold()))
)
.limit({max_items})
.toList()
"""

PROPERTY_QUERY_NEIGHBOR_TPL = """\
g.V().has('{prop}', within({keywords}))
.repeat(
   bothE({edge_labels}).limit({edge_limit}).otherV().dedup()
).times({max_deep}).emit()
.simplePath()
.path()
.by(project('label', 'props')
   .by(label())
   .by(valueMap().by(unfold()))
)
.by(project('label', 'inV', 'outV', 'props')
   .by(label())
   .by(inV().values('{prop}'))
   .by(outV().values('{prop}'))
   .by(valueMap().by(unfold()))
)
.limit({max_items})
.toList()
"""

GREMLIN_GENERATE_EXAMPLE_OPTION_TPL = """\
# Example
Generate gremlin from the following user input.
{example_query}
The generated gremlin is:
```gremlin
{example_gremlin}
```

"""

GREMLIN_GENERATE_TPL = """\
Given the graph schema:
```json
{schema}
```
Given the extracted vertex vid:
{vertices}
Generate gremlin from the following user input.
{query}
The generated gremlin is:"""


class GraphRAGQuery:

    def __init__(
            self,
            max_deep: int = 2,
            max_items: int = 20,
            prop_to_match: Optional[str] = None,
            llm: Optional[BaseLLM] = None,
            embedding: Optional[BaseEmbedding] = None,
    ):
        self._client = PyHugeClient(
            settings.graph_ip,
            settings.graph_port,
            settings.graph_name,
            settings.graph_user,
            settings.graph_pwd,
            settings.graph_space,
        )
        self._max_deep = max_deep
        self._max_items = max_items
        self._prop_to_match = prop_to_match
        self._schema = ""
        self._index_dir = os.path.join(resource_path, "gremlin_examples")
        self._vector_index = VectorIndex.from_index_file(self._index_dir)
        self._llm = llm
        self._embedding = embedding

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # pylint: disable=R0915 (too-many-statements)
        if self._llm is None:
            self._llm = LLMs().get_llm()
        if self._embedding is None:
            self._embedding = Embeddings().get_embedding()
        if self._client is None:
            if isinstance(context.get("graph_client"), PyHugeClient):
                self._client = context["graph_client"]
            else:
                ip = context.get("ip") or "localhost"
                port = context.get("port") or "8080"
                graph = context.get("graph") or "hugegraph"
                user = context.get("user") or "admin"
                pwd = context.get("pwd") or "admin"
                gs = context.get("graphspace") or None
                self._client = PyHugeClient(ip, port, graph, user, pwd, gs)
        assert self._client is not None, "No valid graph to search."

        # 1. Try to perform a query based on the generated gremlin
        context = self._gremlin_generate_query(context)
        # 2. Try to perform a query based on subgraph-search if the previous query failed
        if not context.get("graph_result"):
            context = self._subgraph_query(context)

        # TODO: replace print to log
        verbose = context.get("verbose") or False
        if verbose:
            print("\033[93mKnowledge from Graph:")
            print("\n".join(context["graph_result"]) + "\033[0m")

        return context

    def _gremlin_generate_query(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context["query"]
        vertices = context.get("match_vids")
        query_embedding = context.get("query_embedding")
        if query_embedding is None:
            query_embedding = self._embedding.get_text_embedding(query)

        match_result = self._vector_index.search(query_embedding, top_k=1, dis_threshold=2)
        prompt = ""
        if match_result:
            prompt += GREMLIN_GENERATE_EXAMPLE_OPTION_TPL.format(
                example_query=match_result[0]["query"],
                example_gremlin=match_result[0]["gremlin"]
            )
        else:
            log.warning("No matching example found, generate gremlin with no example.")
        prompt += GREMLIN_GENERATE_TPL.format(
            schema=json.dumps(context["schema"], ensure_ascii=False),
            vertices="\n".join([f"- {vid}" for vid in vertices]),
            query=query
        )

        response = self._llm.generate(prompt=prompt)
        match = re.search("```gremlin.*```", response, re.DOTALL)
        if match:
            gremlin = match.group()[len("```gremlin"):-len("```")]
            log.info("Generated gremlin: %s", gremlin)
            context["gremlin"] = gremlin
            try:
                result = self._client.gremlin().exec(gremlin=gremlin)["data"]
                if result == [None]:
                    result = []
                context["graph_result"] = [json.dumps(item, ensure_ascii=False) for item in result]
                context["graph_context_head"] = (
                    f"The following are graph query result "
                    f"from gremlin query `{gremlin}`.\n"
                )
            except Exception as e:
                log.error(e)
        else:
            log.error("Failed to generate gremlin from the query.")
        return context

    def _subgraph_query(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Extract params from context
        matched_vids = context.get("match_vids")
        if isinstance(context.get("max_deep"), int):
            self._max_deep = context["max_deep"]
        if isinstance(context.get("max_items"), int):
            self._max_items = context["max_items"]
        if isinstance(context.get("prop_to_match"), str):
            self._prop_to_match = context["prop_to_match"]

        # 2. Extract edge_labels from graph schema
        _, edge_labels = self._extract_labels_from_schema()
        edge_labels_str = ",".join("'" + label + "'" for label in edge_labels)
        # TODO: enhance the limit logic later
        edge_limit_amount = len(edge_labels) * 10

        use_id_to_match = self._prop_to_match is None
        if use_id_to_match:
            if not matched_vids:
                return context

            gremlin_query = VERTEX_QUERY_TPL.format(keywords=matched_vids)
            vertexes = self._client.gremlin().exec(gremlin=gremlin_query)["data"]
            log.debug("Vids gremlin query: %s", gremlin_query)

            vertex_knowledge = self._format_graph_from_vertex(query_result=vertexes)
            gremlin_query = VID_QUERY_NEIGHBOR_TPL.format(
                keywords=matched_vids,
                max_deep=self._max_deep,
                edge_labels=edge_labels_str,
                edge_limit=edge_limit_amount,
                max_items=self._max_items,
            )
            log.debug("Kneighbor gremlin query: %s", gremlin_query)
            paths = self._client.gremlin().exec(gremlin=gremlin_query)["data"]

            graph_chain_knowledge, vertex_degree_list, knowledge_with_degree = self._format_graph_query_result(
                query_paths=paths
            )
            if vertex_degree_list:
                vertex_degree_list[0].update(vertex_knowledge)
            else:
                vertex_degree_list.append(vertex_knowledge)
        else:
            # WARN: When will the query enter here?
            keywords = context.get("keywords")
            assert keywords, "No related property(keywords) for graph query."
            keywords_str = ",".join("'" + kw + "'" for kw in keywords)
            gremlin_query = PROPERTY_QUERY_NEIGHBOR_TPL.format(
                prop=self._prop_to_match,
                keywords=keywords_str,
                edge_labels=edge_labels_str,
                edge_limit=edge_limit_amount,
                max_deep=self._max_deep,
                max_items=self._max_items,
            )
            log.warning("Unable to find vid, downgraded to property query, please confirm if it meets expectation.")

            paths: List[Any] = self._client.gremlin().exec(gremlin=gremlin_query)["data"]
            graph_chain_knowledge, vertex_degree_list, knowledge_with_degree = self._format_graph_query_result(
                query_paths=paths
            )

        context["graph_result"] = list(graph_chain_knowledge)
        context["vertex_degree_list"] = [list(vertex_degree) for vertex_degree in vertex_degree_list]
        context["knowledge_with_degree"] = knowledge_with_degree
        context["graph_context_head"] = (
            f"The following are graph knowledge in {self._max_deep} depth, e.g:\n"
            "`vertexA --[links]--> vertexB <--[links]-- vertexC ...`"
            "extracted based on key entities as subject:\n"
        )
        # TODO: set color for ↓ "\033[93mKnowledge from Graph:\033[0m"
        log.debug("Knowledge from Graph:")
        log.debug("\n".join(context["graph_result"]))
        return context

    def _format_graph_from_vertex(self, query_result: List[Any]) -> Set[str]:
        knowledge = set()
        for item in query_result:
            props_str = ", ".join(f"{k}: {v}" for k, v in item["properties"].items())
            node_str = f"{item['id']}{{{props_str}}}"
            knowledge.add(node_str)
        return knowledge

    def _format_graph_query_result(self, query_paths) -> Tuple[Set[str], List[Set[str]], Dict[str, List[str]]]:
        use_id_to_match = self._prop_to_match is None
        subgraph = set()
        subgraph_with_degree = {}
        vertex_degree_list: List[Set[str]] = []
        v_cache: Set[str] = set()
        e_cache: Set[str] = set()

        for path in query_paths:
            # 1. Process each path
            flat_rel, nodes_with_degree = self._process_path(path, use_id_to_match, v_cache, e_cache)
            subgraph.add(flat_rel)
            subgraph_with_degree[flat_rel] = nodes_with_degree
            # 2. Update vertex degree list
            self._update_vertex_degree_list(vertex_degree_list, nodes_with_degree)

        return subgraph, vertex_degree_list, subgraph_with_degree

    def _process_path(self, path: Any, use_id_to_match: bool, v_cache: Set[str],
                      e_cache: Set[str]) -> Tuple[str, List[str]]:
        flat_rel = ""
        raw_flat_rel = path["objects"]
        assert len(raw_flat_rel) % 2 == 1, "The length of raw_flat_rel should be odd."

        node_cache = set()
        prior_edge_str_len = 0
        depth = 0
        nodes_with_degree = []

        for i, item in enumerate(raw_flat_rel):
            if i % 2 == 0:
                # Process each vertex
                flat_rel, prior_edge_str_len, depth = self._process_vertex(
                    item, flat_rel, node_cache, prior_edge_str_len, depth, nodes_with_degree, use_id_to_match,
                    v_cache
                )
            else:
                # Process each edge
                flat_rel, prior_edge_str_len = self._process_edge(
                    item, flat_rel, prior_edge_str_len, raw_flat_rel, i,use_id_to_match, e_cache
                )

        return flat_rel, nodes_with_degree

    def _process_vertex(self, item: Any, flat_rel: str, node_cache: Set[str],
                        prior_edge_str_len: int, depth: int, nodes_with_degree: List[str],
                        use_id_to_match: bool, v_cache: Set[str]) -> Tuple[str, int, int]:
        matched_str = item["id"] if use_id_to_match else item["props"][self._prop_to_match]
        if matched_str in node_cache:
            flat_rel = flat_rel[:-prior_edge_str_len]
            return flat_rel, prior_edge_str_len, depth

        node_cache.add(matched_str)
        props_str = ", ".join(f"{k}: {v}" for k, v in item["props"].items())
        # TODO: we may remove label id or replace with label name
        if matched_str in v_cache:
            node_str = matched_str
        else:
            v_cache.add(matched_str)
            node_str = f"{item['id']}{{{props_str}}}"
        flat_rel += node_str
        nodes_with_degree.append(node_str)
        depth += 1

        return flat_rel, prior_edge_str_len, depth

    def _process_edge(self, item: Any, flat_rel: str, prior_edge_str_len: int,
                      raw_flat_rel: List[Any], i: int, use_id_to_match: bool, e_cache: Set[str]) -> Tuple[str, int]:
        props_str = ", ".join(f"{k}: {v}" for k, v in item["props"].items())
        props_str = f"{{{props_str}}}" if len(props_str) > 0 else ""
        prev_matched_str = raw_flat_rel[i - 1]["id"] if use_id_to_match else (
            raw_flat_rel)[i - 1]["props"][self._prop_to_match]

        if item["label"] in e_cache:
            edge_str = f"{item['label']}"
        else:
            e_cache.add(item["label"])
            edge_str = f"{item['label']}{props_str}"

        if item["outV"] == prev_matched_str:
            edge_str = f" --[{edge_str}]--> "
        else:
            edge_str = f" <--[{edge_str}]-- "

        flat_rel += edge_str
        prior_edge_str_len = len(edge_str)
        return flat_rel, prior_edge_str_len

    def _update_vertex_degree_list(self, vertex_degree_list: List[Set[str]], nodes_with_degree: List[str]) -> None:
        for depth, node_str in enumerate(nodes_with_degree):
            if depth >= len(vertex_degree_list):
                vertex_degree_list.append(set())
            vertex_degree_list[depth].add(node_str)

    def _extract_labels_from_schema(self) -> Tuple[List[str], List[str]]:
        schema = self._get_graph_schema()
        vertex_props_str, edge_props_str = schema.split("\n")[:2]
        # TODO: rename to vertex (also need update in the schema)
        vertex_props_str = vertex_props_str[len("Vertex properties: "):].strip("[").strip("]")
        edge_props_str = edge_props_str[len("Edge properties: "):].strip("[").strip("]")
        vertex_labels = self._extract_label_names(vertex_props_str)
        edge_labels = self._extract_label_names(edge_props_str)
        return vertex_labels, edge_labels

    @staticmethod
    def _extract_label_names(source: str, head: str = "name: ", tail: str = ", ") -> List[str]:
        result = []
        for s in source.split(head):
            end = s.find(tail)
            label = s[:end]
            if label:
                result.append(label)
        return result

    def _get_graph_schema(self, refresh: bool = False) -> str:
        if self._schema and not refresh:
            return self._schema

        schema = self._client.schema()
        vertex_schema = schema.getVertexLabels()
        edge_schema = schema.getEdgeLabels()
        relationships = schema.getRelations()

        self._schema = (
            f"Vertex properties: {vertex_schema}\n"
            f"Edge properties: {edge_schema}\n"
            f"Relationships: {relationships}\n"
        )
        log.debug("Link(Relation): %s", relationships)
        return self._schema
