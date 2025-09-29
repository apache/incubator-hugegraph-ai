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
from typing import Any, Dict, Optional

from PyCGraph import CStatus

from hugegraph_llm.nodes.base_node import BaseNode
from hugegraph_llm.operators.llm_op.gremlin_generate import GremlinGenerateSynthesize
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.config import prompt as prompt_cfg


def _stable_schema_string(state_json: Dict[str, Any]) -> str:
    if "simple_schema" in state_json and state_json["simple_schema"] is not None:
        return json.dumps(state_json["simple_schema"], ensure_ascii=False, sort_keys=True)
    if "schema" in state_json and state_json["schema"] is not None:
        return json.dumps(state_json["schema"], ensure_ascii=False, sort_keys=True)
    return ""


class Text2GremlinNode(BaseNode):
    operator: GremlinGenerateSynthesize

    def node_init(self):
        # Select LLM
        llm = LLMs().get_text2gql_llm()
        # Serialize schema deterministically
        state_json = self.context.to_json()
        schema_str = _stable_schema_string(state_json)
        # Prompt fallback
        gremlin_prompt: Optional[str] = getattr(self.wk_input, "gremlin_prompt", None)
        if gremlin_prompt is None or not str(gremlin_prompt).strip():
            gremlin_prompt = prompt_cfg.gremlin_generate_prompt
        # Keep vertices/properties empty for now
        self.operator = GremlinGenerateSynthesize(
            llm=llm,
            schema=schema_str,
            vertices=None,
            gremlin_prompt=gremlin_prompt,
        )
        return CStatus()

    def operator_schedule(self, data_json: Dict[str, Any]):
        # Ensure query exists in context; return empty if not provided
        query = getattr(self.wk_input, "query", "") or ""
        data_json["query"] = query
        if not query:
            data_json["result"] = ""
            data_json["raw_result"] = ""
            return data_json
        # increase call count for observability
        prev = data_json.get("call_count", 0) or 0
        data_json["call_count"] = prev + 1
        return self.operator.run(data_json)
