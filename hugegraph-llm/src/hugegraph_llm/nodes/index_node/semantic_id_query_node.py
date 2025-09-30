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
from hugegraph_llm.operators.index_op.semantic_id_query import SemanticIdQuery
from hugegraph_llm.models.embeddings.init_embedding import get_embedding
from hugegraph_llm.config import huge_settings, llm_settings
from hugegraph_llm.utils.log import log


class SemanticIdQueryNode(BaseNode):
    """
    Semantic ID query node, responsible for semantic matching based on keywords.
    """

    semantic_id_query: SemanticIdQuery

    def node_init(self):
        """
        Initialize the semantic ID query operator.
        """
        try:
            graph_name = huge_settings.graph_name
            if not graph_name:
                return CStatus(-1, "graph_name is required in wk_input")

            embedding = get_embedding(llm_settings)
            by = self.wk_input.semantic_by or "keywords"
            topk_per_keyword = (
                self.wk_input.topk_per_keyword or huge_settings.topk_per_keyword
            )
            topk_per_query = self.wk_input.topk_per_query or 10
            vector_dis_threshold = (
                self.wk_input.vector_dis_threshold or huge_settings.vector_dis_threshold
            )

            # Initialize the semantic ID query operator
            self.semantic_id_query = SemanticIdQuery(
                embedding=embedding,
                by=by,
                topk_per_keyword=topk_per_keyword,
                topk_per_query=topk_per_query,
                vector_dis_threshold=vector_dis_threshold,
            )

            return super().node_init()
        except Exception as e:
            log.error(f"Failed to initialize SemanticIdQueryNode: {e}")

            return CStatus(-1, f"SemanticIdQueryNode initialization failed: {e}")

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the semantic ID query operation.
        """
        try:
            # Get the query text and keywords from input
            query = data_json.get("query", "")
            keywords = data_json.get("keywords", [])

            if not query and not keywords:
                log.warning("No query text or keywords provided for semantic query")
                return data_json

            # Perform the semantic query
            semantic_result = self.semantic_id_query.run(data_json)

            match_vids = semantic_result.get("match_vids", [])
            log.info(
                f"Semantic query completed, found {len(match_vids)} matching vertex IDs"
            )

            return semantic_result

        except Exception as e:
            log.error(f"Semantic query failed: {e}")
            return data_json
