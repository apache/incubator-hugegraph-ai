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
from pyhugegraph.api.common import HugeModule
from pyhugegraph.api.graph import GraphManager
from pyhugegraph.api.graphs import GraphsManager
from pyhugegraph.api.gremlin import GremlinManager
from pyhugegraph.api.metric import MetricsManager
from pyhugegraph.api.schema import SchemaManager
from pyhugegraph.api.task import TaskManager
from pyhugegraph.api.traverser import TraverserManager
from pyhugegraph.api.variable import VariableManager
from pyhugegraph.api.version import VersionManager
from pyhugegraph.structure.huge_context import HugeContext


def mount(fn):
    attr_name = "_lazy_" + fn.__name__

    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return wrapper


class PyHugeClient(HugeModule):
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
        super().__init__(HugeContext(ip, port, user, pwd, graph, gs, timeout))

    @mount
    def schema(self):
        return SchemaManager(self._ctx)

    @mount
    def gremlin(self):
        return GremlinManager(self._ctx)

    @mount
    def graph(self):
        return GraphManager(self._ctx)

    @mount
    def graphs(self):
        return GraphsManager(self._ctx)

    @mount
    def variable(self):
        return VariableManager(self._ctx)

    @mount
    def auth(self):
        return AuthManager(self._ctx)

    @mount
    def task(self):
        return TaskManager(self._ctx)

    @mount
    def metrics(self):
        return MetricsManager(self._ctx)

    @mount
    def traverser(self):
        return TraverserManager(self._ctx)

    @mount
    def version(self):
        return VersionManager(self._ctx)
