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
from pyhugegraph.utils.huge_decorator import decorator_params, decorator_create
from pyhugegraph.utils.exceptions import CreateError, RemoveError
from pyhugegraph.utils.util import check_if_authorized, check_if_success


class IndexLabel(HugeParamsBase):

    @decorator_params
    def onV(self, vertex_label) -> "IndexLabel":
        self._parameter_holder.set("base_value", vertex_label)
        self._parameter_holder.set("base_type", "VERTEX_LABEL")
        return self

    @decorator_params
    def onE(self, edge_label) -> "IndexLabel":
        self._parameter_holder.set("base_value", edge_label)
        self._parameter_holder.set("base_type", "EDGE_LABEL")
        return self

    @decorator_params
    def by(self, *args) -> "IndexLabel":
        if "fields" not in self._parameter_holder.get_keys():
            self._parameter_holder.set("fields", set())
        s = self._parameter_holder.get_value("fields")
        for item in args:
            s.add(item)
        return self

    @decorator_params
    def secondary(self) -> "IndexLabel":
        self._parameter_holder.set("index_type", "SECONDARY")
        return self

    @decorator_params
    def range(self) -> "IndexLabel":
        self._parameter_holder.set("index_type", "RANGE")
        return self

    @decorator_params
    def search(self) -> "IndexLabel":
        self._parameter_holder.set("index_type", "SEARCH")
        return self

    @decorator_params
    def shard(self) -> "IndexLabel":
        self._parameter_holder.set("index_type", "SHARD")
        return self

    @decorator_params
    def unique(self) -> "IndexLabel":
        self._parameter_holder.set("index_type", "UNIQUE")
        return self

    @decorator_params
    def ifNotExist(self) -> "IndexLabel":
        path = f'schema/indexlabels/{self._parameter_holder.get_value("name")}'
        response = self._sess.request(path)
        if response.status_code == 200 and check_if_authorized(response):
            self._parameter_holder.set("not_exist", False)
        return self

    @decorator_create
    def create(self):
        dic = self._parameter_holder.get_dic()
        data = {}
        data["name"] = dic["name"]
        data["base_type"] = dic["base_type"]
        data["base_value"] = dic["base_value"]
        data["index_type"] = dic["index_type"]
        data["fields"] = list(dic["fields"])
        path = f"schema/indexlabels"
        response = self._sess.request(path, "POST", data=json.dumps(data))
        self.clean_parameter_holder()
        error = CreateError(
            f'CreateError: "create IndexLabel failed", Detail "{str(response.content)}"'
        )
        if check_if_success(response, error):
            return f'create IndexLabel success, Deatil: "{str(response.content)}"'
        return None

    @decorator_params
    def remove(self):
        name = self._parameter_holder.get_value("name")
        path = f"schema/indexlabels/{name}"
        response = self._sess.request(path, "DELETE")
        self.clean_parameter_holder()
        error = RemoveError(
            f'RemoveError: "remove IndexLabel failed", Detail "{str(response.content)}"'
        )
        if check_if_success(response, error):
            return f'remove IndexLabel success, Deatil: "{str(response.content)}"'
        return None
