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

import requests
import traceback

from abc import ABC
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HGraphContext:
    ip: str
    port: str
    username: str
    password: str
    graph_name: str
    graphspace: Optional[str] = None
    timeout: int = 10
    api_version: str = field(default="v1", init=False)

    def __post_init__(self):

        if self.graphspace is not None:
            self.api_version = "v3"

        else:
            try:
                response = requests.get(
                    f"http://{self.ip}:{self.port}/versions", timeout=1
                )
                version = response.json()["versions"]["version"]
                print(f"Retrieved API version information from the server: {version}.")

                if version == "v3":
                    self.graphspace = "DEFAULT"
                    print(
                        f"graph space is not set, default value 'DEFAULT' will be used."
                    )

                self.api_version = version

            except Exception as e:
                traceback.print_exception(e)
                self.api_version = "v1"
                print(
                    "Failed to retrieve API version information from the server, reverting to default v1."
                )


class HGraphBaseModel(ABC):
    def __init__(self, ctx: HGraphContext):
        self._ctx = ctx
