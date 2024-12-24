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
from typing import Any, Dict

from hugegraph_llm.config import huge_settings
from hugegraph_llm.utils.log import log
from pyhugegraph.client import PyHugeClient


class GremlinExecute:
    def __init__(self):
        self._client = PyHugeClient(
            huge_settings.graph_ip,
            huge_settings.graph_port,
            huge_settings.graph_name,
            huge_settings.graph_user,
            huge_settings.graph_pwd,
            huge_settings.graph_space,
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        gremlin = context.get("gremlin_result")

        try:
            result = self._client.gremlin().exec(gremlin=gremlin)["data"]
            if result == [None]:
                result = []
            context["graph_result"] = [json.dumps(item, ensure_ascii=False) for item in result]
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error(e)

        if context.get("graph_result"):
            context["graph_result_flag"] = 1
            context["graph_context_head"] = f"The following are graph query result from gremlin query `{gremlin}`.\n"
        return context
