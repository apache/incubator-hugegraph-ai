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

from typing import Dict, Any
from hugegraph_llm.config import llm_settings
from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.index_op.vector_index_query import VectorIndexQuery
from hugegraph_llm.models.embeddings.init_embedding import get_embedding
from hugegraph_llm.utils.log import log


class VectorQueryNode(BaseNode):
    """
    Vector query node, responsible for retrieving relevant documents from the vector index
    """

    operator: VectorIndexQuery

    def node_init(self):
        """
        Initialize the vector query operator
        """
        try:
            # 从 wk_input 中读取用户配置参数
            embedding = get_embedding(llm_settings)
            max_items = (
                self.wk_input.max_items if self.wk_input.max_items is not None else 3
            )

            self.operator = VectorIndexQuery(embedding=embedding, topk=max_items)
            return super().node_init()
        except Exception as e:
            log.error(f"Failed to initialize VectorQueryNode: {e}")
            from PyCGraph import CStatus

            return CStatus(-1, f"VectorQueryNode initialization failed: {e}")

    def operator_schedule(self, data_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the vector query operation
        """
        try:
            # Get the query text from input
            query = data_json.get("query", "")
            if not query:
                log.warning("No query text provided for vector query")
                return data_json

            # Perform the vector query
            result = self.operator.run({"query": query})

            # Update the state
            data_json.update(result)
            log.info(
                f"Vector query completed, found {len(result.get('vector_result', []))} results"
            )

            return data_json

        except Exception as e:
            log.error(f"Vector query failed: {e}")
            return data_json
