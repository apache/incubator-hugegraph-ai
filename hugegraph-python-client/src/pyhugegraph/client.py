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

from typing import Optional

from pyhugegraph.api.auth import AuthManager
from pyhugegraph.api.graph import GraphManager
from pyhugegraph.api.graphs import GraphsManager
from pyhugegraph.api.gremlin import GremlinManager
from pyhugegraph.api.metric import MetricsManager
from pyhugegraph.api.schema import SchemaManager
from pyhugegraph.api.task import TaskManager
from pyhugegraph.api.traverser import TraverserManager
from pyhugegraph.api.variable import VariableManager
from pyhugegraph.api.version import VersionManager
from pyhugegraph.utils.huge_config import HGraphConfig
from pyhugegraph.utils.huge_requests import HGraphSession


class PyHugeClient:
    def __init__(
        self,
        ip: str,
        port: str,
        graph: str,
        user: str,
        pwd: str,
        timeout: int = 10,
        gs: Optional[str] = None,
    ):
        self._cfg = HGraphConfig(ip, port, user, pwd, graph, gs, timeout)

    @staticmethod
    def __router(fn):
        attr_name = "_lazy_" + fn.__name__

        def wrapper(self: "PyHugeClient"):
            if not hasattr(self, attr_name):
                session = HGraphSession(self._cfg)
                setattr(self, attr_name, fn(self)(session))
            return getattr(self, attr_name)

        return wrapper

    @__router
    def schema(self) -> SchemaManager:
        return SchemaManager

    @__router
    def gremlin(self) -> GremlinManager:
        return GremlinManager

    @__router
    def graph(self) -> GraphManager:
        return GraphManager

    @__router
    def graphs(self) -> GraphsManager:
        return GraphsManager

    @__router
    def variable(self) -> VariableManager:
        return VariableManager

    @__router
    def auth(self) -> AuthManager:
        return AuthManager

    @__router
    def task(self) -> TaskManager:
        return TaskManager

    @__router
    def metrics(self) -> MetricsManager:
        return MetricsManager

    @__router
    def traverser(self) -> TraverserManager:
        return TraverserManager

    @__router
    def version(self) -> VersionManager:
        return VersionManager

    def __repr__(self) -> str:
        return f"{self._cfg}"
