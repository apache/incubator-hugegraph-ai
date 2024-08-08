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

from pyhugegraph.api.common import HugeParamsBase
from pyhugegraph.structure.gremlin_data import GremlinData
from pyhugegraph.structure.response_data import ResponseData
from pyhugegraph.utils.exceptions import NotFoundError
from pyhugegraph.utils import huge_router as router
from pyhugegraph.utils.util import check_if_success


class GremlinManager(HugeParamsBase):

    @router.http("POST", "/gremlin")
    def exec(self, gremlin):
        gremlin_data = GremlinData(gremlin)
        if self._sess._cfg.gs_supported:
            gremlin_data.aliases = {
                "graph": f"{self._sess._cfg.graphspace}-{self._sess._cfg.graph_name}",
                "g": f"__g_{self._sess._cfg.graphspace}-{self._sess._cfg.graph_name}",
            }
        else:
            gremlin_data.aliases = {
                "graph": f"{self._sess._cfg.graph_name}",
                "g": f"__g_{self._sess._cfg.graph_name}",
            }
        response = self._invoke_request(data=gremlin_data.to_json())
        error = NotFoundError(f"Gremlin can't get results: {str(response.content)}")
        if check_if_success(response, error):
            return ResponseData(json.loads(response.content)).result
        return ""
