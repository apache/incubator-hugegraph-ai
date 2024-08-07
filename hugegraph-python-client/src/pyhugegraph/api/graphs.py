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

from pyhugegraph.utils.constants import Constants
from pyhugegraph.utils.exceptions import NotFoundError
from pyhugegraph.utils.util import check_if_success


class GraphsManager(HugeParamsBase):

    def get_all_graphs(self):
        uri = '/graphs'

        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""

    def get_version(self):
        uri = f"/versions"
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""

    def get_graph_info(self):
        uri = ""
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""

    def clear_graph_all_data(self):
        uri = f"clear?confirm_message={Constants.CONFORM_MESSAGE}"
        response = self._sess.delete(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""

    def get_graph_config(self):
        uri = "conf"
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""
