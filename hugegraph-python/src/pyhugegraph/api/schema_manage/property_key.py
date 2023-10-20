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

import json

import requests

from pyhugegraph.api.common import HugeParamsBase
from pyhugegraph.utils.exceptions import CreateError, UpdateError, RemoveError
from pyhugegraph.utils.huge_decorator import decorator_params, decorator_create
from pyhugegraph.utils.util import check_if_success, check_if_authorized


class PropertyKey(HugeParamsBase):
    def __init__(self, graph_instance, session):
        super().__init__(graph_instance)
        self.session = session

    @decorator_params
    def asInt(self):
        self._parameter_holder.set("data_type", "INT")
        return self

    @decorator_params
    def asText(self):
        self._parameter_holder.set("data_type", "TEXT")
        return self

    @decorator_params
    def asDouble(self):
        self._parameter_holder.set("data_type", "DOUBLE")
        return self

    @decorator_params
    def asDate(self):
        self._parameter_holder.set("data_type", "DATE")
        return self

    @decorator_params
    def valueSingle(self):
        self._parameter_holder.set("cardinality", "SINGLE")
        return self

    @decorator_params
    def valueList(self):
        self._parameter_holder.set("cardinality", "LIST")
        return self

    @decorator_params
    def valueSet(self):
        self._parameter_holder.set("cardinality", "SET")
        return self

    @decorator_params
    def calcMax(self):
        self._parameter_holder.set("aggregate_type", "MAX")
        return self

    @decorator_params
    def calcMin(self):
        self._parameter_holder.set("aggregate_type", "MIN")
        return self

    @decorator_params
    def calcSum(self):
        self._parameter_holder.set("aggregate_type", "SUM")
        return self

    @decorator_params
    def calcOld(self):
        self._parameter_holder.set("aggregate_type", "OLD")
        return self

    @decorator_params
    def userdata(self, *args):
        user_data = self._parameter_holder.get_value("user_data")
        if not user_data:
            self._parameter_holder.set("user_data", dict())
            user_data = self._parameter_holder.get_value("user_data")
        i = 0
        while i < len(args):
            user_data[args[i]] = args[i + 1]
            i += 2
        return self

    def ifNotExist(self):
        url = f'{self._host}/graphs/{self._graph_name}/schema/propertykeys/{self._parameter_holder.get_value("name")}'
        response = self.session.get(url, auth=self._auth, headers=self._headers)
        if response.status_code == 200 and check_if_authorized(response):
            self._parameter_holder.set("not_exist", False)
        return self

    @decorator_create
    def create(self):
        dic = self._parameter_holder.get_dic()
        property_keys = {
            "name": dic["name"]
        }
        if "data_type" in dic:
            property_keys["data_type"] = dic["data_type"]
        if "cardinality" in dic:
            property_keys["cardinality"] = dic["cardinality"]
        url = f'{self._host}/graphs/{self._graph_name}/schema/propertykeys'
        response = self.session.post(url, data=json.dumps(property_keys), auth=self._auth, headers=self._headers)
        self.clean_parameter_holder()
        if check_if_success(response, CreateError(
                'CreateError: "create PropertyKey failed", Detail: {}'.format(response.content))):
            return 'create PropertyKey success, Detail: {}'.format(response.content)

    @decorator_params
    def append(self):
        property_name = self._parameter_holder.get_value("name")
        user_data = self._parameter_holder.get_value("user_data")
        if not user_data:
            user_data = dict()
        data = {
            "name": property_name,
            "user_data": user_data
        }

        url = f'{self._host}/graphs/{self._graph_name}/schema/propertykeys/{property_name}/?action=append'
        response = self.session.put(url, data=json.dumps(data), auth=self._auth, headers=self._headers)
        self.clean_parameter_holder()
        if check_if_success(response, UpdateError(
                'UpdateError: "append PropertyKey failed", Detail: {}'.format(response.content))):
            return 'append PropertyKey success, Detail: {}'.format(response.content)

    @decorator_params
    def eliminate(self):
        property_name = self._parameter_holder.get_value("name")
        user_data = self._parameter_holder.get_value("user_data")
        if not user_data:
            user_data = dict()
        data = {
            "name": property_name,
            "user_data": user_data
        }

        url = f'{self._host}/graphs/{self._graph_name}/schema/propertykeys/{property_name}/?action=eliminate'
        response = self.session.put(url, data=json.dumps(data), auth=self._auth, headers=self._headers)
        self.clean_parameter_holder()
        error = UpdateError(f'UpdateError: "eliminate PropertyKey failed", Detail: {str(response.content)}')
        if check_if_success(response, error):
            return 'eliminate PropertyKey success, Detail: {}'.format(str(response.content))

    @decorator_params
    def remove(self):
        dic = self._parameter_holder.get_dic()
        url = f'{self._host}/graphs/{self._graph_name}/schema/propertykeys/{dic["name"]}'
        response = self.session.delete(url)
        self.clean_parameter_holder()
        if check_if_success(response, RemoveError(
                'RemoveError: "delete PropertyKey failed", Detail: {}'.format(str(response.content)))):
            return 'delete PropertyKey success, Detail: {}'.format(dic["name"])
