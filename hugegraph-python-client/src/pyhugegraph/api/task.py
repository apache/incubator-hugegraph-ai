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
from pyhugegraph.utils.exceptions import NotFoundError
from pyhugegraph.utils.huge_requests import HugeSession
from pyhugegraph.utils.util import check_if_success


class TaskManager(HugeParamsBase):
    def __init__(self, graph_instance):
        super().__init__(graph_instance)
        self.session = self.set_session(HugeSession.new_session())

    def set_session(self, session):
        self.session = session
        return session

    def close(self):
        if self.session:
            self.session.close()

    def list_tasks(self, status=None, limit=None):
        url = f"{self._host}/graphs/{self._graph_name}/tasks"
        params = {}
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        response = self.session.get(
            url,
            params=params,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        check_if_success(response, NotFoundError(response.content))
        return response.json()

    def get_task(self, task_id):
        url = f"{self._host}/graphs/{self._graph_name}/tasks/{task_id}"
        response = self.session.get(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        check_if_success(response, NotFoundError(response.content))
        return response.json()

    def delete_task(self, task_id):
        url = f"{self._host}/graphs/{self._graph_name}/tasks/{task_id}"
        response = self.session.delete(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        check_if_success(response, NotFoundError(response.content))
        return response.status_code

    def cancel_task(self, task_id):
        url = f"{self._host}/graphs/{self._graph_name}/tasks/{task_id}?action=cancel"
        response = self.session.put(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        check_if_success(response, NotFoundError(response.content))
        return response.json()
