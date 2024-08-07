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
from pyhugegraph.utils.util import check_if_success


class TaskManager(HugeParamsBase):

    def list_tasks(self, status=None, limit=None):
        uri = 'tasks'
        params = {}
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        response = self._sess.get(uri, params=params)
        check_if_success(response, NotFoundError(response.content))
        return response.json()

    def get_task(self, task_id):
        uri = f"tasks/{task_id}"
        response = self._sess.get(uri)
        check_if_success(response, NotFoundError(response.content))
        return response.json()

    def delete_task(self, task_id):
        uri = f"tasks/{task_id}"
        response = self._sess.delete(uri)
        check_if_success(response, NotFoundError(response.content))
        return response.status_code

    def cancel_task(self, task_id):
        uri = f"tasks/{task_id}?action=cancel"
        response = self._sess.put(uri)
        check_if_success(response, NotFoundError(response.content))
        return response.json()
