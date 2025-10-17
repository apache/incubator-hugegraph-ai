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

from typing import Any, Dict

from PyCGraph import CStatus

from hugegraph_llm.config import index_settings
from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.index_op.gremlin_example_index_query import (
    GremlinExampleIndexQuery,
)
from hugegraph_llm.models.embeddings.init_embedding import Embeddings


class GremlinExampleIndexQueryNode(BaseNode):
    operator: GremlinExampleIndexQuery

    def node_init(self):
        # Lazy import to avoid circular dependency
        from hugegraph_llm.utils.vector_index_utils import get_vector_index_class  # pylint: disable=import-outside-toplevel
        
        # Build operator (index lazy-loading handled in operator)
        vector_index = get_vector_index_class(index_settings.cur_vector_index)
        embedding = Embeddings().get_embedding()
        example_num = getattr(self.wk_input, "example_num", None)
        if not isinstance(example_num, int):
            example_num = 2
        # Clamp to [0, 10]
        example_num = max(0, min(10, example_num))
        self.operator = GremlinExampleIndexQuery(vector_index=vector_index, embedding=embedding, num_examples=example_num)
        return CStatus()

    def operator_schedule(self, data_json: Dict[str, Any]):
        # Ensure query is present in context; degrade gracefully if empty
        query = getattr(self.wk_input, "query", "") or ""
        data_json["query"] = query
        if not query:
            data_json["match_result"] = []
            return data_json
        # Operator.run writes match_result into context
        return self.operator.run(data_json)
