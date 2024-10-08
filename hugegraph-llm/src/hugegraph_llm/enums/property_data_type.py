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


from enum import Enum


class PropertyDataType(Enum):
    BOOLEAN = "BOOLEAN"
    BYTE = "BYTE"
    INT = "INT"
    LONG = "LONG"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    TEXT = "TEXT"
    BLOB = "BLOB"
    DATE = "DATE"
    UUID = "UUID"
    DEFAULT = TEXT


def default_value_map(data_type: str):
    return {
        "BOOLEAN": False,
        "BYTE": 0,
        "INT": 0,
        "LONG": 0,
        "FLOAT": 0.0,
        "DOUBLE": 0.0,
        "TEXT": "",
        "BLOB": "",
        "DATE": "2000-01-01",
        "UUID": "",
    }[data_type]
