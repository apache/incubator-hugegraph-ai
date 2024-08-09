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
from pyhugegraph.utils import huge_router as router
from pyhugegraph.utils.util import check_if_success


class AuthManager(HugeParamsBase):

    @router.http("GET", "auth/users")
    def list_users(self, limit=None):
        params = {"limit": limit} if limit is not None else {}
        response = self._invoke_request(params=params)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return []

    @router.http("POST", "auth/users")
    def create_user(self, user_name, user_password, user_phone=None, user_email=None):
        data = {
            "user_name": user_name,
            "user_password": user_password,
            "user_phone": user_phone,
            "user_email": user_email,
        }
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("DELETE", "auth/users/{user_id}")
    def delete_user(self, user_id):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            if response.status_code != 204:
                return response.json()
        return {}

    @router.http("PUT", "auth/users/{user_id}")
    def modify_user(
        self,
        user_id,
        user_name=None,
        user_password=None,
        user_phone=None,
        user_email=None,
    ):
        data = {
            "user_name": user_name,
            "user_password": user_password,
            "user_phone": user_phone,
            "user_email": user_email,
        }
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("GET", "auth/users/{user_id}")
    def get_user(self, user_id):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("GET", "auth/groups")
    def list_groups(self, limit=None):
        params = {"limit": limit} if limit is not None else {}
        response = self._invoke_request(params=params)
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return []

    @router.http("POST", "auth/groups")
    def create_group(self, group_name, group_description=None):
        data = {"group_name": group_name, "group_description": group_description}
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("DELETE", "auth/groups/{group_id}")
    def delete_group(self, group_id):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            if response.status_code != 204:
                return response.json()
        return {}

    @router.http("PUT", "auth/groups/{group_id}")
    def modify_group(self, group_id, group_name=None, group_description=None):
        data = {"group_name": group_name, "group_description": group_description}
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("GET", "auth/groups/{group_id}")
    def get_group(self, group_id):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("POST", "auth/accesses")
    def grant_accesses(self, group_id, target_id, access_permission):
        data = {
            "group": group_id,
            "target": target_id,
            "access_permission": access_permission,
        }
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("DELETE", "auth/accesses/{access_id}")
    def revoke_accesses(self, access_id):
        response = self._invoke_request()
        check_if_success(response, NotFoundError(response.content))

    @router.http("PUT", "auth/accesses/{access_id}")
    def modify_accesses(self, access_id, access_description):
        # The permission of access can\'t be updated
        data = {"access_description": access_description}
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("GET", "auth/accesses/{access_id}")
    def get_accesses(self, access_id):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("GET", "auth/accesses")
    def list_accesses(self):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("POST", "auth/targets")
    def create_target(self, target_name, target_graph, target_url, target_resources):
        data = {
            "target_name": target_name,
            "target_graph": target_graph,
            "target_url": target_url,
            "target_resources": target_resources,
        }
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("DELETE", "auth/targets/{target_id}")
    def delete_target(self, target_id):
        response = self._invoke_request()
        check_if_success(response, NotFoundError(response.content))

    @router.http("PUT", "auth/targets/{target_id}")
    def update_target(
        self, target_id, target_name, target_graph, target_url, target_resources
    ):
        data = {
            "target_name": target_name,
            "target_graph": target_graph,
            "target_url": target_url,
            "target_resources": target_resources,
        }
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("GET", "auth/targets/{target_id}")
    def get_target(self, target_id, response=None):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("GET", "auth/targets")
    def list_targets(self):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("POST", "auth/belongs")
    def create_belong(self, user_id, group_id):
        data = {"user": user_id, "group": group_id}
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("DELETE", "auth/belongs/{belong_id}")
    def delete_belong(self, belong_id):
        response = self._invoke_request()
        check_if_success(response, NotFoundError(response.content))

    @router.http("PUT", "auth/belongs/{belong_id}")
    def update_belong(self, belong_id, description):
        data = {"belong_description": description}
        response = self._invoke_request(data=json.dumps(data))
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("GET", "auth/belongs/{belong_id}")
    def get_belong(self, belong_id):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}

    @router.http("GET", "auth/belongs")
    def list_belongs(self):
        response = self._invoke_request()
        if check_if_success(response, NotFoundError(response.content)):
            return response.json()
        return {}
