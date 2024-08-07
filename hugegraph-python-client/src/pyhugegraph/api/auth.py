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
from pyhugegraph.utils.util import check_if_success


class AuthManager(HugeParamsBase):

    def list_users(self, limit=None):
        uri = "auth/users"
        params = {}
        if limit is not None:
            params["limit"] = limit
        response = self._sess.get(
            uri,
            params=params,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return []

    def create_user(self, user_name, user_password, user_phone=None, user_email=None):
        uri = "auth/users"
        data = {
            "user_name": user_name,
            "user_password": user_password,
            "user_phone": user_phone,
            "user_email": user_email,
        }
        response = self._sess.post(
            uri,
            data=json.dumps(data),
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def delete_user(self, user_id):
        uri = f"auth/users/{user_id}"
        response = self._sess.delete(
            uri,
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
        uri = f"auth/users/{user_id}"
        data = {
            "user_name": user_name,
            "user_password": user_password,
            "user_phone": user_phone,
            "user_email": user_email,
        }
        response = self._sess.put(
            uri,
            data=json.dumps(data),
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_user(self, user_id):
        uri = f"auth/users/{user_id}"
        response = self._sess.get(
            uri,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def list_groups(self, limit=None):
        uri = "auth/groups"
        params = {}
        if limit is not None:
            params["limit"] = limit
        response = self._sess.get(
            uri,
            params=params,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return []

    def create_group(self, group_name, group_description=None):
        uri = "auth/groups"
        data = {"group_name": group_name, "group_description": group_description}
        response = self._sess.post(
            uri,
            data=json.dumps(data),
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def delete_group(self, group_id):
        uri = f"auth/groups/{group_id}"
        response = self._sess.delete(
            uri,
        )
        if check_if_success(response, NotFoundError(response.content)):
            if response.status_code != 204:
                return response.json()
        return {}

    def modify_group(self, group_id, group_name=None, group_description=None):
        uri = f"auth/groups/{group_id}"
        data = {"group_name": group_name, "group_description": group_description}
        response = self._sess.put(
            uri,
            data=json.dumps(data),
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_group(self, group_id):
        uri = f"auth/groups/{group_id}"
        response = self._sess.get(
            uri,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def grant_accesses(self, group_id, target_id, access_permission):
        uri = "auth/accesses"
        data = {
            "group": group_id,
            "target": target_id,
            "access_permission": access_permission,
        }
        response = self._sess.post(
            uri,
            data=json.dumps(data),
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def revoke_accesses(self, access_id):
        uri = f"auth/accesses/{access_id}"
        response = self._sess.delete(
            uri,
        )
        check_if_success(response, NotFoundError(response.content))

    def modify_accesses(self, access_id, access_description):
        uri = f"auth/accesses/{access_id}"
        # The permission of access can\'t be updated
        data = {"access_description": access_description}
        response = self._sess.put(
            uri,
            data=json.dumps(data),
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_accesses(self, access_id):
        uri = f"auth/accesses/{access_id}"
        response = self._sess.get(
            uri,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def list_accesses(self):
        uri = f"auth/accesses"
        response = self._sess.get(
            uri,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def create_target(self, target_name, target_graph, target_url, target_resources):
        uri = "auth/targets"
        data = {
            "target_name": target_name,
            "target_graph": target_graph,
            "target_url": target_url,
            "target_resources": target_resources,
        }
        response = self._sess.post(
            uri,
            data=json.dumps(data),
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def delete_target(self, target_id):
        uri = f"auth/targets/{target_id}"
        response = self._sess.delete(
            uri,
        )
        check_if_success(response, NotFoundError(response.content))

    def update_target(
        self, target_id, target_name, target_graph, target_url, target_resources
    ):
        uri = f"auth/targets/{target_id}"
        data = {
            "target_name": target_name,
            "target_graph": target_graph,
            "target_url": target_url,
            "target_resources": target_resources,
        }
        response = self._sess.put(
            uri,
            data=json.dumps(data),
            auth=self._auth,
            headers=self._headers,
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_target(self, target_id, response=None):
        uri = f"auth/targets/{target_id}"
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def list_targets(self):
        uri = f"auth/targets"
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def create_belong(self, user_id, group_id):
        uri = f"auth/belongs"
        data = {"user": user_id, "group": group_id}
        response = self._sess.post(
            uri,
            data=json.dumps(data),
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def delete_belong(self, belong_id):
        uri = f"auth/belongs/{belong_id}"
        response = self._sess.delete(
            uri,
        )
        check_if_success(response, NotFoundError(response.content))

    def update_belong(self, belong_id, description):
        uri = f"auth/belongs/{belong_id}"
        data = {"belong_description": description}
        response = self._sess.put(
            uri,
            data=json.dumps(data),
        )
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def get_belong(self, belong_id):
        uri = f"auth/belongs/{belong_id}"
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    def list_belongs(self):
        uri = f"auth/belongs"
        response = self._sess.get(uri)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}
