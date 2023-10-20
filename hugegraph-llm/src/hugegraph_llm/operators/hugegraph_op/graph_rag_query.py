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

from pyhugegraph.client import PyHugeClient


class GraphRAGQuery:
    ID_RAG_GREMLIN_QUERY_TEMPL = (
        "g.V().hasId({keywords}).as('subj')"
        ".repeat("
        "   bothE({edge_labels}).as('rel').otherV().as('obj')"
        ").times({max_deep})"
        ".path()"
        ".by(project('label', 'id', 'props')"
        "   .by(label())"
        "   .by(id())"
        "   .by(valueMap().by(unfold()))"
        ")"
        ".by(project('label', 'inV', 'outV', 'props')"
        "   .by(label())"
        "   .by(inV().id())"
        "   .by(outV().id())"
        "   .by(valueMap().by(unfold()))"
        ")"
        ".limit({max_items})"
        ".toList()"
    )
    PROP_RAG_GREMLIN_QUERY_TEMPL = (
        "g.V().has('{prop}', within({keywords})).as('subj')"
        ".repeat("
        "   bothE({edge_labels}).as('rel').otherV().as('obj')"
        ").times({max_deep})"
        ".path()"
        ".by(project('label', 'props')"
        "   .by(label())"
        "   .by(valueMap().by(unfold()))"
        ")"
        ".by(project('label', 'inV', 'outV', 'props')"
        "   .by(label())"
        "   .by(inV().values('{prop}'))"
        "   .by(outV().values('{prop}'))"
        "   .by(valueMap().by(unfold()))"
        ")"
        ".limit({max_items})"
        ".toList()"
    )

    def __init__(
            self,
            client: Optional[PyHugeClient] = None,
            max_deep: int = 2,
            max_items: int = 30,
            prop_to_match: Optional[str] = None,
    ):
        self._client = client
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
                port = context.get("port") or 8080
                graph = context.get("graph") or "hugegraph"
                user = context.get("user") or "admin"
                pwd = context.get("pwd") or "admin"
                self._client = PyHugeClient(
                    ip=ip, port=port, graph=graph, user=user, pwd=pwd
                )
        assert self._client is not None, "No graph for query."

        keywords = context.get("keywords")
        assert keywords is not None, "No keywords for query."

        if isinstance(context.get("max_deep"), int):
            self._max_deep = context["max_deep"]
        if isinstance(context.get("max_items"), int):
            self._max_items = context["max_items"]
        if isinstance(context.get("prop_to_match"), str):
            self._prop_to_match = context["prop_to_match"]

        node_labels, edge_labels = self._extract_labels_from_schema()
        edge_labels_str = ",".join("'" + label + "'" for label in edge_labels)

        use_id_to_match = self._prop_to_match is None

        if not use_id_to_match:
            keywords_str = ",".join("'" + kw + "'" for kw in keywords)
            rag_gremlin_query_template = self.PROP_RAG_GREMLIN_QUERY_TEMPL
            rag_gremlin_query = rag_gremlin_query_template.format(
                prop=self._prop_to_match,
                keywords=keywords_str,
                max_deep=self._max_deep,
                max_items=self._max_items,
                edge_labels=edge_labels_str,
            )
        else:
            id_format = self._get_graph_id_format()
            if id_format == "STRING":
                keywords_str = ",".join("'" + kw + "'" for kw in keywords)
            else:
                raise RuntimeError("Unsupported ID format for Graph RAG.")

            rag_gremlin_query_template = self.ID_RAG_GREMLIN_QUERY_TEMPL
            rag_gremlin_query = rag_gremlin_query_template.format(
                keywords=keywords_str,
                max_deep=self._max_deep,
                max_items=self._max_items,
                edge_labels=edge_labels_str,
            )

        result: List[Any] = self._client.gremlin().exec(gremlin=rag_gremlin_query)["data"]
        knowledge: Set[str] = self._format_knowledge_from_query_result(
            query_result=result
        )

        context["synthesize_context_body"] = list(knowledge)
        context["synthesize_context_head"] = (
            f"The following are knowledge sequence in max depth {self._max_deep} "
            f"in the form of directed graph like:\n"
            "`subject -[predicate]-> object <-[predicate_next_hop]- object_next_hop ...` "
            "extracted based on key entities as subject:"
        )

        verbose = context.get("verbose") or False
        if verbose:
            print(f"\033[93mKNOWLEDGE FROM GRAPH:")
            print("\n".join(rel for rel in context["synthesize_context_body"]) + "\033[0m")

        return context

    def _format_knowledge_from_query_result(
            self,
            query_result: List[Any],
    ) -> Set[str]:
        use_id_to_match = self._prop_to_match is None
        knowledge = set()
        for line in query_result:
            flat_rel = ""
            raw_flat_rel = line["objects"]
            assert len(raw_flat_rel) % 2 == 1
            node_cache = set()
            prior_edge_str_len = 0
            for i, item in enumerate(raw_flat_rel):
                if i % 2 == 0:
                    matched_str = (
                        item["id"]
                        if use_id_to_match
                        else item["props"][self._prop_to_match]
                    )
                    if matched_str in node_cache:
                        flat_rel = flat_rel[:-prior_edge_str_len]
                        break
                    node_cache.add(matched_str)
                    props_str = ", ".join(f"{k}: {v}" for k, v in item["props"].items())
                    node_str = f"{item['label']}{{{props_str}}}"
                    flat_rel += node_str
                    if flat_rel in knowledge:
                        knowledge.remove(flat_rel)
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
        return knowledge

    def _extract_labels_from_schema(self) -> Tuple[List[str], List[str]]:
        schema = self._get_graph_schema()
        node_props_str, edge_props_str = schema.split("\n")[:2]
        node_props_str = (
            node_props_str[len("Node properties: "):].strip("[").strip("]")
        )
        edge_props_str = (
            edge_props_str[len("Edge properties: "):].strip("[").strip("]")
        )
        node_labels = self._extract_label_names(node_props_str)
        edge_labels = self._extract_label_names(edge_props_str)
        return node_labels, edge_labels

    @staticmethod
    def _extract_label_names(
            s: str,
            head: str = "name: ",
            tail: str = ", ",
    ) -> List[str]:
        result = []
        for s in s.split(head):
            end = s.find(tail)
            label = s[:end]
            if label:
                result.append(label)
        return result

    def _get_graph_id_format(self) -> str:
        sample = self._client.gremlin().exec("g.V().limit(1)")["data"]
        if len(sample) == 0:
            return "EMPTY"
        else:
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
