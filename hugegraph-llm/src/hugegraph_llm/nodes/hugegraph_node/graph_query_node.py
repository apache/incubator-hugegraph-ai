#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from PyCGraph import CStatus
from typing import Dict, Any
from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.hugegraph_op.graph_rag_query import GraphRAGQuery
from hugegraph_llm.config import huge_settings, prompt
from hugegraph_llm.utils.log import log


class GraphQueryNode(BaseNode):
    """
    Graph query node, responsible for retrieving relevant information from the graph database.
    """

    graph_rag_query: GraphRAGQuery

    def node_init(self):
        """
        Initialize the graph query operator.
        """
        try:
            graph_name = huge_settings.graph_name
            if not graph_name:
                return CStatus(-1, "graph_name is required in wk_input")

            max_deep = self.wk_input.max_deep or 2
            max_graph_items = (
                self.wk_input.max_graph_items or huge_settings.max_graph_items
            )
            max_v_prop_len = self.wk_input.max_v_prop_len or 2048
            max_e_prop_len = self.wk_input.max_e_prop_len or 256
            prop_to_match = self.wk_input.prop_to_match
            num_gremlin_generate_example = self.wk_input.gremlin_tmpl_num or -1
            gremlin_prompt = (
                self.wk_input.gremlin_prompt or prompt.gremlin_generate_prompt
            )

            # Initialize GraphRAGQuery operator
            self.graph_rag_query = GraphRAGQuery(
                max_deep=max_deep,
                max_graph_items=max_graph_items,
                max_v_prop_len=max_v_prop_len,
                max_e_prop_len=max_e_prop_len,
                prop_to_match=prop_to_match,
                num_gremlin_generate_example=num_gremlin_generate_example,
                gremlin_prompt=gremlin_prompt,
            )

            return super().node_init()
        except Exception as e:
            log.error(f"Failed to initialize GraphQueryNode: {e}")

            return CStatus(-1, f"GraphQueryNode initialization failed: {e}")

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the graph query operation.
        """
        try:
            # Get the query text from input
            query = data_json.get("query", "")

            if not query:
                log.warning("No query text provided for graph query")
                return data_json

            # Execute the graph query (assuming schema and semantic query have been completed in previous nodes)
            graph_result = self.graph_rag_query.run(data_json)
            data_json.update(graph_result)

            log.info(
                f"Graph query completed, found {len(data_json.get('graph_result', []))} results"
            )

            return data_json

        except Exception as e:
            log.error(f"Graph query failed: {e}")
            return data_json
