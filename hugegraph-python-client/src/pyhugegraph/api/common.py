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
from pyhugegraph.utils.constants import Constants


class ParameterHolder:
    def __init__(self):
        self._dic = {}

    def set(self, key, value):
        self._dic[key] = value

    def get_value(self, key):
        if key not in self._dic:
            return None
        else:
            return self._dic[key]

    def get_dic(self):
        return self._dic

    def get_keys(self):
        return self._dic.keys()


class HugeParamsBase:
    def __init__(self, graph_instance):
        self._graph_instance = graph_instance
        self._ip = graph_instance.ip
        self._port = graph_instance.port
        self._user = graph_instance.user_name
        self._pwd = graph_instance.passwd
        self._host = "http://{}:{}".format(graph_instance.ip, graph_instance.port)
        self._auth = (graph_instance.user_name, graph_instance.passwd)
        self._graph_name = graph_instance.graph_name
        self._parameter_holder = None
        self._headers = {
            'Content-Type': Constants.HEADER_CONTENT_TYPE
        }
        self._timeout = graph_instance.timeout

    def add_parameter(self, key, value):
        self._parameter_holder.set(key, value)
        return

    def get_parameter_holder(self):
        return self._parameter_holder

    def create_parameter_holder(self):
        self._parameter_holder = ParameterHolder()

    def clean_parameter_holder(self):
        self._parameter_holder = None
