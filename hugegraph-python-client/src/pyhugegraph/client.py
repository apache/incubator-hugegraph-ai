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
from pyhugegraph.structure.graph_instance import GraphInstance


class PyHugeClient(HugeParamsBase):
    def __init__(self, ip, port, graph, user, pwd, timeout=10):
        self._graph_instance = GraphInstance(ip, port, graph, user, pwd, timeout)
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

    def schema(self):
        if self._schema:
            return self._schema
        self._schema = SchemaManager(self._graph_instance)
        return self._schema

    def gremlin(self):
        if self._gremlin:
            return self._gremlin
        self._gremlin = GremlinManager(self._graph_instance)
        return self._gremlin

    def graph(self):
        if self._graph:
            return self._graph
        self._graph = GraphManager(self._graph_instance)
        return self._graph

    def graphs(self):
        if self._graphs:
            return self._graphs
        self._graphs = GraphsManager(self._graph_instance)
        return self._graphs

    def variable(self):
        if self._variable:
            return self._variable
        self._variable = VariableManager(self._graph_instance)
        return self._variable

    def auth(self):
        if self._auth:
            return self._auth
        self._auth = AuthManager(self._graph_instance)
        return self._auth

    def task(self):
        if self._task:
            return self._task
        self._task = TaskManager(self._graph_instance)
        return self._task

    def metrics(self):
        if self._metrics:
            return self._metrics
        self._metrics = MetricsManager(self._graph_instance)
        return self._metrics

    def traverser(self):
        if self._traverser:
            return self._traverser
        self._traverser = TraverserManager(self._graph_instance)
        return self._traverser
