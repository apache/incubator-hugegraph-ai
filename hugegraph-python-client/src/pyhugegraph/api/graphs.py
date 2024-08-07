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


from pyhugegraph.api.common import HugeParamsBase
from pyhugegraph.utils import huge_router as router
from pyhugegraph.utils.exceptions import NotFoundError
from pyhugegraph.utils.util import check_if_success


class GraphsManager(HugeParamsBase):

    @router.http("GET", "/graphs")
    def get_all_graphs(self):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""

    @router.http("GET", "/versions")
    def get_version(self):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""

    @router.http("GET", "")
    def get_graph_info(self):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""

    @router.http("DELETE", f"clear?confirm_message=I%27m+sure+to+delete+all+data")
    def clear_graph_all_data(self):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""

    @router.http("GET", "conf")
    def get_graph_config(self):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""
