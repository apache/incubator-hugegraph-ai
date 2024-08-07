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
from pyhugegraph.structure.base_model import HGraphContext, HGraphBaseModel


def add_router(fn):
    attr_name = "_lazy_" + fn.__name__

    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return wrapper


class PyHugeClient(HGraphBaseModel):
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
        super().__init__(HGraphContext(ip, port, user, pwd, graph, gs, timeout))

    @add_router
    def schema(self):
        return SchemaManager(self._ctx)

    @add_router
    def gremlin(self):
        return GremlinManager(self._ctx)

    @add_router
    def graph(self):
        return GraphManager(self._ctx)

    @add_router
    def graphs(self):
        return GraphsManager(self._ctx)

    @add_router
    def variable(self):
        return VariableManager(self._ctx)

    @add_router
    def auth(self):
        return AuthManager(self._ctx)

    @add_router
    def task(self):
        return TaskManager(self._ctx)

    @add_router
    def metrics(self):
        return MetricsManager(self._ctx)

    @add_router
    def traverser(self):
        return TraverserManager(self._ctx)

    @add_router
    def version(self):
        return VersionManager(self._ctx)
