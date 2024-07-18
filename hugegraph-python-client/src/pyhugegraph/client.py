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
from pyhugegraph.api.auth import AuthManager
from pyhugegraph.api.common import HugeParamsBase
from pyhugegraph.api.graph import GraphManager
from pyhugegraph.api.graphs import GraphsManager
from pyhugegraph.api.gremlin import GremlinManager
from pyhugegraph.api.metric import MetricsManager
from pyhugegraph.api.schema import SchemaManager
from pyhugegraph.api.task import TaskManager
from pyhugegraph.api.traverser import TraverserManager
from pyhugegraph.api.variable import VariableManager
from pyhugegraph.api.version import VersionManager
from pyhugegraph.structure.graph_instance import GraphInstance


class PyHugeClient(HugeParamsBase):
    def __init__(self, ip, port, graph, user, pwd, timeout=10, graphspace=None):
        self._graph_instance = GraphInstance(ip, port, graph, user, pwd, timeout, graphspace)
        super().__init__(self._graph_instance)
        self._schema = None
        self._graph = None
        self._graphs = None
        self._gremlin = None
        self._variable = None
        self._auth = None
        self._task = None
        self._metrics = None
        self._traverser = None
        self._version = None

    def schema(self):
        self._schema = self._schema or SchemaManager(self._graph_instance)
        return self._schema

    def gremlin(self):
        self._gremlin = self._gremlin or GremlinManager(self._graph_instance)
        return self._gremlin

    def graph(self):
        self._graph = self._graph or GraphManager(self._graph_instance)
        return self._graph

    def graphs(self):
        self._graphs = self._graphs or GraphsManager(self._graph_instance)
        return self._graphs

    def variable(self):
        self._variable = self._variable or VariableManager(self._graph_instance)
        return self._variable

    def auth(self):
        self._auth = self._auth or AuthManager(self._graph_instance)
        return self._auth

    def task(self):
        self._task = self._task or TaskManager(self._graph_instance)
        return self._task

    def metrics(self):
        self._metrics = self._metrics or MetricsManager(self._graph_instance)
        return self._metrics

    def traverser(self):
        self._traverser = self._traverser or TraverserManager(self._graph_instance)
        return self._traverser

    def version(self):
        self._version = self._version or VersionManager(self._graph_instance)
        return self._version
