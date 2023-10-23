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
from pyhugegraph.utils.exceptions import NotFoundError
from pyhugegraph.utils.huge_requests import HugeSession
from pyhugegraph.utils.util import check_if_success


class VariableManager(HugeParamsBase):

    def __init__(self, graph_instance):
        super().__init__(graph_instance)
        self.session = self.set_session(HugeSession.new_session())

    def set_session(self, session):
        self.session = session
        return session

    def close(self):
        if self.session:
            self.session.close()

    def set(self, key, value):
        url = f'{self._host}/graphs/{self._graph_name}/variables/{key}'
        data = {'data': value}

        response = self.session.put(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get(self, key):
        url = f'{self._host}/graphs/{self._graph_name}/variables/{key}'

        response = self.session.get(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def all(self):
        url = f'{self._host}/graphs/{self._graph_name}/variables'

        response = self.session.get(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def remove(self, key):
        url = f'{self._host}/graphs/{self._graph_name}/variables/{key}'

        response = self.session.delete(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout
        )
        check_if_success(response, NotFoundError(response.content))
