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

import requests

from src.api.common import HugeParamsBase
from src.utils.exceptions import CreateError, UpdateError, RemoveError
from src.utils.huge_decorator import decorator_params, decorator_create
from src.utils.util import check_if_success, check_if_authorized


class VertexLabel(HugeParamsBase):
    def __init__(self, graph_instance, session):
        super().__init__(graph_instance)
        self.session = session

    @decorator_params
    def useAutomaticId(self):
        self._parameter_holder.set("id_strategy", "AUTOMATIC")
        return self

    @decorator_params
    def useCustomizeStringId(self):
        self._parameter_holder.set("id_strategy", "CUSTOMIZE_STRING")
        return self

    @decorator_params
    def useCustomizeNumberId(self):
        self._parameter_holder.set("id_strategy", "CUSTOMIZE_NUMBER")
        return self

    @decorator_params
    def usePrimaryKeyId(self):
        self._parameter_holder.set("id_strategy", "PRIMARY_KEY")
        return self

    @decorator_params
    def properties(self, *args):
        self._parameter_holder.set("properties", list(args))
        return self

    @decorator_params
    def primaryKeys(self, *args):
        self._parameter_holder.set("primary_keys", list(args))
        return self

    @decorator_params
    def nullableKeys(self, *args):
        self._parameter_holder.set("nullable_keys", list(args))
        return self

    @decorator_params
    def enableLabelIndex(self, flag):
        self._parameter_holder.set("enable_label_index", flag)
        return self

    @decorator_params
    def userdata(self, *args):
        if "user_data" not in self._parameter_holder.get_keys():
            self._parameter_holder.set('user_data', dict())
        user_data = self._parameter_holder.get_value('user_data')
        i = 0
        while i < len(args):
            user_data[args[i]] = args[i + 1]
            i += 2
        return self

    def ifNotExist(self):
        url = f'{self._host}/graphs/{self._graph_name}/schema/vertexlabels/{self._parameter_holder.get_value("name")}'

        response = self.session.get(url, auth=self._auth, headers=self._headers)
        if response.status_code == 200 and check_if_authorized(response):
            self._parameter_holder.set("not_exist", False)
        return self

    @decorator_create
    def create(self):
        dic = self._parameter_holder.get_dic()
        key_list = ["name", "id_strategy", "primary_keys", "nullable_keys", "index_labels",
                    "properties", "enable_label_index", "user_data"]
        data = {}
        for key in key_list:
            if key in dic:
                data[key] = dic[key]
        url = f'{self._host}/graphs/{self._graph_name}/schema/vertexlabels'
        response = self.session.post(url, data=json.dumps(data), auth=self._auth, headers=self._headers)
        self.clean_parameter_holder()
        error = CreateError('CreateError: "create VertexLabel failed", Detail: "{}"'
                            .format(str(response.content)))
        if check_if_success(response, error):
            return 'create VertexLabel success, Detail: "{}"'.format(str(response.content))

    @decorator_params
    def append(self):
        dic = self._parameter_holder.get_dic()
        properties = dic['properties'] if "properties" in dic else []
        nullable_keys = dic['nullable_keys'] if "nullable_keys" in dic else []
        user_data = dic['user_data'] if 'user_data' in dic else {}
        url = f'{self._host}/graphs/{self._graph_name}/schema/vertexlabels/{dic["name"]}?action=append'

        data = {
            "name": dic["name"],
            "properties": properties,
            "nullable_keys": nullable_keys,
            "user_data": user_data
        }
        response = self.session.put(url, data=json.dumps(data), auth=self._auth, headers=self._headers)
        self.clean_parameter_holder()
        error = UpdateError('UpdateError: "append VertexLabel failed", Detail: "{}"'.
                            format(str(response.content)))
        if check_if_success(response, error):
            return 'append VertexLabel success, Detail: "{}"'.format(str(response.content))

    @decorator_params
    def remove(self):
        name = self._parameter_holder.get_value("name")
        url = f'{self._host}/graphs/{self._graph_name}/schema/vertexlabels/{name}'
        response = self.session.delete(url, auth=self._auth, headers=self._headers)
        self.clean_parameter_holder()
        error = RemoveError('RemoveError: "remove VertexLabel failed", Detail: "{}"'.
                            format(str(response.content)))
        if check_if_success(response, error):
            return 'remove VertexLabel success, Detail: "{}"'.format(str(response.content))

    @decorator_params
    def eliminate(self):
        name = self._parameter_holder.get_value("name")
        url = f'{self._host}/graphs/{self._graph_name}/schema/vertexlabels/{name}/?action=eliminate'

        dic = self._parameter_holder.get_dic()
        user_data = dic['user_data'] if 'user_data' in dic else {}
        data = {
            "name": self._parameter_holder.get_value("name"),
            "user_data": user_data,
        }
        response = self.session.put(url, data=json.dumps(data), auth=self._auth, headers=self._headers)
        error = UpdateError('UpdateError: "eliminate VertexLabel failed", Detail: "{}"'.
                            format(str(response.content)))
        if check_if_success(response, error):
            return 'eliminate VertexLabel success, Detail: "{}"'.format(str(response.content))
