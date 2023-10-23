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
from pyhugegraph.utils.exceptions import CreateError, UpdateError, RemoveError
from pyhugegraph.utils.huge_decorator import decorator_params, decorator_create
from pyhugegraph.utils.util import check_if_success, check_if_authorized


class EdgeLabel(HugeParamsBase):
    def __init__(self, graph_instance, session):
        super().__init__(graph_instance)
        self.session = session

    @decorator_params
    def link(self, source_label, target_label):
        self._parameter_holder.set("source_label", source_label)
        self._parameter_holder.set("target_label", target_label)
        return self

    @decorator_params
    def sourceLabel(self, source_label):
        self._parameter_holder.set("source_label", source_label)
        return self

    @decorator_params
    def targetLabel(self, target_label):
        self._parameter_holder.set("target_label", target_label)
        return self

    @decorator_params
    def userdata(self, *args):
        if not self._parameter_holder.get_value("user_data"):
            self._parameter_holder.set("user_data", {})
        user_data = self._parameter_holder.get_value("user_data")
        i = 0
        while i < len(args):
            user_data[args[i]] = args[i + 1]
            i += 2
        return self

    @decorator_params
    def properties(self, *args):
        self._parameter_holder.set("properties", list(args))
        return self

    @decorator_params
    def singleTime(self):
        self._parameter_holder.set("frequency", "SINGLE")
        return self

    @decorator_params
    def multiTimes(self):
        self._parameter_holder.set("frequency", "MULTIPLE")
        return self

    @decorator_params
    def sortKeys(self, *args):
        self._parameter_holder.set("sort_keys", list(args))
        return self

    @decorator_params
    def nullableKeys(self, *args):
        nullable_keys = set(args)
        self._parameter_holder.set("nullable_keys", list(nullable_keys))
        return self

    @decorator_params
    def ifNotExist(self):
        url = f'{self._host}/graphs/{self._graph_name}/schema/edgelabels/{self._parameter_holder.get_value("name")}'

        response = self.session.get(url, auth=self._auth, headers=self._headers)
        if response.status_code == 200 and check_if_authorized(response):
            self._parameter_holder.set("not_exist", False)
        return self

    @decorator_create
    def create(self):
        dic = self._parameter_holder.get_dic()
        data = {}
        keys = [
            "name",
            "source_label",
            "target_label",
            "nullable_keys",
            "properties",
            "enable_label_index",
            "sort_keys",
            "user_data",
            "frequency",
        ]
        for key in keys:
            if key in dic:
                data[key] = dic[key]
        url = f"{self._host}/graphs/{self._graph_name}/schema/edgelabels"
        response = self.session.post(
            url, data=json.dumps(data), auth=self._auth, headers=self._headers
        )
        self.clean_parameter_holder()
        error = CreateError(
            f'CreateError: "create EdgeLabel failed", Detail:  "{str(response.content)}"'
        )
        if check_if_success(response, error):
            return f'create EdgeLabel success, Detail: "{str(response.content)}"'

    @decorator_params
    def remove(self):
        url = f'{self._host}/graphs/{self._graph_name}/schema/edgelabels/{self._parameter_holder.get_value("name")}'
        response = self.session.delete(url, auth=self._auth, headers=self._headers)
        self.clean_parameter_holder()
        error = RemoveError(
            f'RemoveError: "remove EdgeLabel failed", Detail:  "{str(response.content)}"'
        )
        if check_if_success(response, error):
            return f'remove EdgeLabel success, Detail: "{str(response.content)}"'

    @decorator_params
    def append(self):
        dic = self._parameter_holder.get_dic()
        data = {}
        keys = ["name", "nullable_keys", "properties", "user_data"]
        for key in keys:
            if key in dic:
                data[key] = dic[key]

        url = (
            f'{self._host}/graphs/{self._graph_name}/schema/edgelabels/{data["name"]}?action=append'
        )
        response = self.session.put(
            url, data=json.dumps(data), auth=self._auth, headers=self._headers
        )
        self.clean_parameter_holder()
        error = UpdateError(
            f'UpdateError: "append EdgeLabel failed", Detail: "{str(response.content)}"'
        )
        if check_if_success(response, error):
            return f'append EdgeLabel success, Detail: "{str(response.content)}"'

    @decorator_params
    def eliminate(self):
        name = self._parameter_holder.get_value("name")
        user_data = (
            self._parameter_holder.get_value("user_data")
            if self._parameter_holder.get_value("user_data")
            else {}
        )
        url = f"{self._host}/graphs/{self._graph_name}/schema/edgelabels/{name}?action=eliminate"
        data = {"name": name, "user_data": user_data}
        response = self.session.put(
            url, data=json.dumps(data), auth=self._auth, headers=self._headers
        )
        self.clean_parameter_holder()
        error = UpdateError(
            f'UpdateError: "eliminate EdgeLabel failed", Detail: "{str(response.content)}"'
        )
        if check_if_success(response, error):
            return f'eliminate EdgeLabel success, Detail: "{str(response.content)}"'
