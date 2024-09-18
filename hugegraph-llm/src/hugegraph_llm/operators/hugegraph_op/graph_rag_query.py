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


import re
from typing import Any, Dict, Optional, List, Set, Tuple

from hugegraph_llm.config import settings
from hugegraph_llm.utils.log import log
from pyhugegraph.client import PyHugeClient

VERTEX_QUERY_TPL = "g.V({keywords}).as('subj').toList()"

# TODO: we could use a simpler query (like kneighbor-api to get the edges)
# TODO: use dedup() to filter duplicate paths
ID_QUERY_NEIGHBOR_TPL = """
g.V({keywords}).as('subj')
.repeat(
   bothE({edge_labels}).as('rel').otherV().as('obj')
).times({max_deep})
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

PROPERTY_QUERY_NEIGHBOR_TPL = """
g.V().has('{prop}', within({keywords})).as('subj')
.repeat(
   bothE({edge_labels}).as('rel').otherV().as('obj')
).times({max_deep})
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


class GraphRAGQuery:

    def __init__(self, max_deep: int = 2, max_items: int = 30, prop_to_match: Optional[str] = None):
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

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
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

        keywords = context.get("keywords")
        match_vids = context.get("match_vids")

        if isinstance(context.get("max_deep"), int):
            self._max_deep = context["max_deep"]
        if isinstance(context.get("max_items"), int):
            self._max_items = context["max_items"]
        if isinstance(context.get("prop_to_match"), str):
            self._prop_to_match = context["prop_to_match"]

        _, edge_labels = self._extract_labels_from_schema()
        edge_labels_str = ",".join("'" + label + "'" for label in edge_labels)

        use_id_to_match = self._prop_to_match is None
        if use_id_to_match:
            if not match_vids:
                return context

            gremlin_query = VERTEX_QUERY_TPL.format(keywords=match_vids)
            result: List[Any] = self._client.gremlin().exec(gremlin=gremlin_query)["data"]
            log.debug(f"Vids query: {gremlin_query}")

            vertex_knowledge = self._format_graph_from_vertex(query_result=result)
            gremlin_query = ID_QUERY_NEIGHBOR_TPL.format(
                keywords=match_vids,
                max_deep=self._max_deep,
                max_items=self._max_items,
                edge_labels=edge_labels_str,
            )
            log.debug(f"Kneighbor query: {gremlin_query}")

            result: List[Any] = self._client.gremlin().exec(gremlin=gremlin_query)["data"]
            graph_chain_knowledge, vertex_degree_list, knowledge_with_degree = self._format_graph_from_query_result(
                query_result=result
            )
            graph_chain_knowledge.update(vertex_knowledge)
            if vertex_degree_list:
                vertex_degree_list[0].update(vertex_knowledge)
            else:
                vertex_degree_list.append(vertex_knowledge)
        else:
            # WARN: When will the query enter here?
            assert keywords, "No related property(keywords) for graph query."
            keywords_str = ",".join("'" + kw + "'" for kw in keywords)
            gremlin_query = PROPERTY_QUERY_NEIGHBOR_TPL.format(
                prop=self._prop_to_match,
                keywords=keywords_str,
                max_deep=self._max_deep,
                max_items=self._max_items,
                edge_labels=edge_labels_str,
            )
            log.warning("Unable to find vid, downgraded to property query, please confirm if it meets expectation.")

            result: List[Any] = self._client.gremlin().exec(gremlin=gremlin_query)["data"]
            graph_chain_knowledge, vertex_degree_list, knowledge_with_degree = self._format_graph_from_query_result(
                query_result=result
            )

        context["graph_result"] = list(graph_chain_knowledge)
        context["vertex_degree_list"] = [list(vertex_degree) for vertex_degree in vertex_degree_list]
        context["knowledge_with_degree"] = knowledge_with_degree
        context["graph_context_head"] = (
            f"The following are knowledge sequence in max depth {self._max_deep} "
            f"in the form of directed graph like:\n"
            "`subject -[predicate]-> object <-[predicate_next_hop]- object_next_hop ...`"
            "extracted based on key entities as subject:\n"
        )

        # TODO: replace print to log
        verbose = context.get("verbose") or False
        if verbose:
            print("\033[93mKnowledge from Graph:")
            print("\n".join(chain for chain in context["graph_result"]) + "\033[0m")

        return context

    def _format_graph_from_vertex(self, query_result: List[Any]) -> Set[str]:
        knowledge = set()
        for item in query_result:
            props_str = ", ".join(f"{k}: {v}" for k, v in item["properties"].items())
            node_str = f"{item['id']}{{{props_str}}}"
            knowledge.add(node_str)
        return knowledge

    def _format_graph_from_query_result(
        self, query_result: List[Any]
    ) -> Tuple[Set[str], List[Set[str]], Dict[str, List[str]]]:
        use_id_to_match = self._prop_to_match is None
        knowledge = set()
        knowledge_with_degree = {}
        vertex_degree_list: List[Set[str]] = []
        for line in query_result:
            flat_rel = ""
            raw_flat_rel = line["objects"]
            assert len(raw_flat_rel) % 2 == 1
            node_cache = set()
            prior_edge_str_len = 0
            depth = 0
            nodes_with_degree = []
            for i, item in enumerate(raw_flat_rel):
                if i % 2 == 0:
                    matched_str = item["id"] if use_id_to_match else item["props"][self._prop_to_match]
                    if matched_str in node_cache:
                        flat_rel = flat_rel[:-prior_edge_str_len]
                        break
                    node_cache.add(matched_str)
                    props_str = ", ".join(f"{k}: {v}" for k, v in item["props"].items())
                    node_str = f"{item['id']}{{{props_str}}}"
                    flat_rel += node_str
                    nodes_with_degree.append(node_str)
                    if flat_rel in knowledge:
                        knowledge.remove(flat_rel)
                        knowledge_with_degree.pop(flat_rel)
                    if depth >= len(vertex_degree_list):
                        vertex_degree_list.append(set())
                    vertex_degree_list[depth].add(node_str)
                    depth += 1
                else:
                    props_str = ", ".join(f"{k}: {v}" for k, v in item["props"].items())
                    props_str = f"{{{props_str}}}" if len(props_str) > 0 else ""
                    prev_matched_str = (
                        raw_flat_rel[i - 1]["id"]
                        if use_id_to_match
                        else raw_flat_rel[i - 1]["props"][self._prop_to_match]
                    )
                    if item["outV"] == prev_matched_str:
                        edge_str = f" -[{item['label']}{props_str}]-> "
                    else:
                        edge_str = f" <-[{item['label']}{props_str}]- "
                    flat_rel += edge_str
                    prior_edge_str_len = len(edge_str)
            knowledge.add(flat_rel)
            knowledge_with_degree[flat_rel] = nodes_with_degree
        return knowledge, vertex_degree_list, knowledge_with_degree

    def _extract_labels_from_schema(self) -> Tuple[List[str], List[str]]:
        schema = self._get_graph_schema()
        node_props_str, edge_props_str = schema.split("\n")[:2]
        node_props_str = node_props_str[len("Node properties: "):].strip("[").strip("]")
        edge_props_str = edge_props_str[len("Edge properties: "):].strip("[").strip("]")
        node_labels = self._extract_label_names(node_props_str)
        edge_labels = self._extract_label_names(edge_props_str)
        return node_labels, edge_labels

    @staticmethod
    def _extract_label_names(source: str, head: str = "name: ", tail: str = ", ") -> List[str]:
        result = []
        for s in source.split(head):
            end = s.find(tail)
            label = s[:end]
            if label:
                result.append(label)
        return result

    def _get_graph_id_format(self) -> str:
        sample = self._client.gremlin().exec("g.V().limit(1)")["data"]
        if len(sample) == 0:
            return "EMPTY"
        sample_id = sample[0]["id"]
        if isinstance(sample_id, int):
            return "INT"
        if isinstance(sample_id, str):
            if re.match(r"^\d+:.*", sample_id):
                return "INT:STRING"
            return "STRING"
        return "UNKNOWN"

    def _get_graph_schema(self, refresh: bool = False) -> str:
        if self._schema and not refresh:
            return self._schema

        schema = self._client.schema()
        vertex_schema = schema.getVertexLabels()
        edge_schema = schema.getEdgeLabels()
        relationships = schema.getRelations()

        self._schema = (
            f"Node properties: {vertex_schema}\n"
            f"Edge properties: {edge_schema}\n"
            f"Relationships: {relationships}\n"
        )
        return self._schema
