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

class HugeGraph:
    """HugeGraph wrapper for graph operations"""

    def __init__(
        self,
        username: str = "default",
        password: str = "default",
        address: str = "127.0.0.1",
        port: int = 8081,
        graph: str = "hugegraph"
    ) -> None:
        """Create a new HugeGraph wrapper instance."""
        try:
            from pyhugegraph.client import PyHugeClient
        except ImportError:
            raise ValueError(
                "Please install HugeGraph Python client first: "
                "`pip3 install hugegraph-python-client`"
            )

        self.username = username
        self.password = password
        self.address = address
        self.port = port
        self.graph = graph
        self.client = PyHugeClient(address, port, user=username, pwd=password, graph=graph)
        self.schema = ""

    def exec(self, query) -> str:
        """Returns the schema of the HugeGraph database"""
        return self.client.gremlin().exec(query)
