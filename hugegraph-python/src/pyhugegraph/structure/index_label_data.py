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


class IndexLabelData:
    def __init__(self, dic):
        self.__id = dic["id"] if "id" in dic else None
        self.__base_type = dic["base_type"] if "base_type" in dic else None
        self.__base_value = dic["base_value"] if "base_value" in dic else None
        self.__name = dic["name"] if "name" in dic else None
        self.__fields = dic["fields"] if "fields" in dic else None
        self.__index_type = dic["index_type"] if "index_type" in dic else None

    @property
    def id(self):
        return self.__id

    @property
    def baseType(self):
        return self.__base_type

    @property
    def baseValue(self):
        return self.__base_value

    @property
    def name(self):
        return self.__name

    @property
    def fields(self):
        return self.__fields

    @property
    def indexType(self):
        return self.__index_type

    def __repr__(self):
        res = "index_name: {}, base_value: {}, base_type: {}, fields: [], index_type: {}"\
            .format(self.__name, self.__base_value, self.__base_type, self.__fields,
                    self.__index_type)
        return res
