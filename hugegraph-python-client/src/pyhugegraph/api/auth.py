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


class AuthManager(HugeParamsBase):
    def __init__(self, graph_instance):
        super().__init__(graph_instance)
        self.__session = HugeSession.new_session()

    def close(self):
        if self.__session:
            self.__session.close()

    def list_users(self, limit=None):
        url = f"{self._host}/graphs/{self._graph_name}/auth/users"
        params = {}
        if limit is not None:
            params["limit"] = limit
        response = self.__session.get(
            url,
            params=params,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return []

    def create_user(self, user_name, user_password, user_phone=None, user_email=None):
        url = f"{self._host}/graphs/{self._graph_name}/auth/users"
        data = {
            "user_name": user_name,
            "user_password": user_password,
            "user_phone": user_phone,
            "user_email": user_email,
        }
        response = self.__session.post(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def delete_user(self, user_id):
        url = f"{self._host}/graphs/{self._graph_name}/auth/users/{user_id}"
        response = self.__session.delete(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            if response.status_code != 204:
                return response.json()
        return {}

    def modify_user(
        self,
        user_id,
        user_name=None,
        user_password=None,
        user_phone=None,
        user_email=None,
    ):
        url = f"{self._host}/graphs/{self._graph_name}/auth/users/{user_id}"
        data = {
            "user_name": user_name,
            "user_password": user_password,
            "user_phone": user_phone,
            "user_email": user_email,
        }
        response = self.__session.put(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_user(self, user_id):
        url = f"{self._host}/graphs/{self._graph_name}/auth/users/{user_id}"
        response = self.__session.get(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def list_groups(self, limit=None):
        url = f"{self._host}/graphs/{self._graph_name}/auth/groups"
        params = {}
        if limit is not None:
            params["limit"] = limit
        response = self.__session.get(
            url,
            params=params,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return []

    def create_group(self, group_name, group_description=None):
        url = f"{self._host}/graphs/{self._graph_name}/auth/groups"
        data = {"group_name": group_name, "group_description": group_description}
        response = self.__session.post(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def delete_group(self, group_id):
        url = f"{self._host}/graphs/{self._graph_name}/auth/groups/{group_id}"
        response = self.__session.delete(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            if response.status_code != 204:
                return response.json()
        return {}

    def modify_group(self, group_id, group_name=None, group_description=None):
        url = f"{self._host}/graphs/{self._graph_name}/auth/groups/{group_id}"
        data = {"group_name": group_name, "group_description": group_description}
        response = self.__session.put(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_group(self, group_id):
        url = f"{self._host}/graphs/{self._graph_name}/auth/groups/{group_id}"
        response = self.__session.get(
            url,
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def grant_accesses(self, group_id, target_id, access_permission):
        url = f"{self._host}/graphs/{self._graph_name}/auth/accesses"
        data = {
            "group": group_id,
            "target": target_id,
            "access_permission": access_permission,
        }
        response = self.__session.post(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def revoke_accesses(self, access_id):
        url = f"{self._host}/graphs/{self._graph_name}/auth/accesses/{access_id}"
        response = self.__session.delete(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        check_if_success(response, NotFoundError(response.content))

    def modify_accesses(self, access_id, access_description):
        url = f"{self._host}/graphs/{self._graph_name}/auth/accesses/{access_id}"
        # The permission of access can\'t be updated
        data = {"access_description": access_description}
        response = self.__session.put(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_accesses(self, access_id):
        url = f"{self._host}/graphs/{self._graph_name}/auth/accesses/{access_id}"
        response = self.__session.get(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def list_accesses(self):
        url = f"{self._host}/graphs/{self._graph_name}/auth/accesses"
        response = self.__session.get(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def create_target(self, target_name, target_graph, target_url, target_resources):
        url = f"{self._host}/graphs/{self._graph_name}/auth/targets"
        data = {
            "target_name": target_name,
            "target_graph": target_graph,
            "target_url": target_url,
            "target_resources": target_resources,
        }
        response = self.__session.post(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def delete_target(self, target_id):
        url = f"{self._host}/graphs/{self._graph_name}/auth/targets/{target_id}"
        response = self.__session.delete(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        check_if_success(response, NotFoundError(response.content))

    def update_target(self, target_id, target_name, target_graph, target_url, target_resources):
        url = f"{self._host}/graphs/{self._graph_name}/auth/targets/{target_id}"
        data = {
            "target_name": target_name,
            "target_graph": target_graph,
            "target_url": target_url,
            "target_resources": target_resources,
        }
        response = self.__session.put(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_target(self, target_id):
        url = f"{self._host}/graphs/{self._graph_name}/auth/targets/{target_id}"
        response = self.__session.get(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def list_targets(self):
        url = f"{self._host}/graphs/{self._graph_name}/auth/targets"
        response = self.__session.get(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def create_belong(self, user_id, group_id):
        url = f"{self._host}/graphs/{self._graph_name}/auth/belongs"
        data = {"user": user_id, "group": group_id}
        response = self.__session.post(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def delete_belong(self, belong_id):
        url = f"{self._host}/graphs/{self._graph_name}/auth/belongs/{belong_id}"
        response = self.__session.delete(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        check_if_success(response, NotFoundError(response.content))

    def update_belong(self, belong_id, description):
        url = f"{self._host}/graphs/{self._graph_name}/auth/belongs/{belong_id}"
        data = {"belong_description": description}
        response = self.__session.put(
            url,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
            timeout=self._timeout,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_belong(self, belong_id):
        url = f"{self._host}/graphs/{self._graph_name}/auth/belongs/{belong_id}"
        response = self.__session.get(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def list_belongs(self):
        url = f"{self._host}/graphs/{self._graph_name}/auth/belongs"
        response = self.__session.get(
            url, auth=self._auth, headers=self._headers, timeout=self._timeout
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}
