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
# KIND, either expresponses or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from pyhugegraph.api.common import HugeParamsBase

from pyhugegraph.utils.constants import Constants
from pyhugegraph.utils.exceptions import NotFoundError
from pyhugegraph.utils.huge_requests import HugeSession
from pyhugegraph.utils.util import check_if_success


class GraphsManager(HugeParamsBase):
    def __init__(self, graph_instance):
        super().__init__(graph_instance)
        self.session = self.set_session(HugeSession.new_session())

    def set_session(self, session):
        self.session = session
        return session

    def close(self):
        if self.session:
            self.session.close()

    def get_all_graphs(self):
        url = f"{self._host}/graphs"
        response = self.session.get(url, auth=self._auth, headers=self._headers)
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""

    def get_version(self):
        url = f"{self._host}/versions"
        response = self.session.get(url, auth=self._auth, headers=self._headers)
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""

    def get_graph_info(self):
        url = f"{self._host}/graphs/{self._graph_name}"
        response = self.session.get(url, auth=self._auth, headers=self._headers)
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""

    def clear_graph_all_data(self):
        url = f"{self._host}/graphs/{self._graph_name}/clear?confirm_message=" \
              f"{Constants.CONFORM_MESSAGE}"
        response = self.session.delete(url, auth=self._auth, headers=self._headers)
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""

    def get_graph_config(self):
        url = self._host + "/graphs" + "/" + self._graph_name + "/conf"
        response = self.session.get(url, auth=self._auth, headers=self._headers)
        if check_if_success(response, NotFoundError(response.content)):
            return str(response.content)
        return ""
