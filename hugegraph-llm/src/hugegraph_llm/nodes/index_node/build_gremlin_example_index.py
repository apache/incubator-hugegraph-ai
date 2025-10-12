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

from hugegraph_llm.config import llm_settings
from hugegraph_llm.models.embeddings.init_embedding import get_embedding
from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.index_op.build_gremlin_example_index import (
    BuildGremlinExampleIndex,
)
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class BuildGremlinExampleIndexNode(BaseNode):
    build_gremlin_example_index_op: BuildGremlinExampleIndex
    context: WkFlowState = None
    wk_input: WkFlowInput = None

    def node_init(self):
        if self.wk_input.examples is not None:
            examples = self.wk_input.examples
        else:
            return CStatus(-1, "examples is required in BuildGremlinExampleIndexNode")
        self.build_gremlin_example_index_op = BuildGremlinExampleIndex(
            get_embedding(llm_settings), examples
        )
        return super().node_init()

    def operator_schedule(self, data_json):
        return self.build_gremlin_example_index_op.run(data_json)
