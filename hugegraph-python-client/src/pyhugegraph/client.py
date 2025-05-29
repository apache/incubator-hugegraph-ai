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

from typing import Any, Callable, Optional, TypeVar

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
from pyhugegraph.utils.log import log

T = TypeVar("T")


def manager_builder(fn: Callable[[Any, "HGraphSession"], T]) -> Callable[[Any], T]:
    attr_name = "_lazy_" + fn.__name__

    def wrapper(self: "PyHugeClient") -> T:
        if not hasattr(self, attr_name):
            session = HGraphSession(self.cfg)
            setattr(self, attr_name, fn(self)(session))
        return getattr(self, attr_name)

    return wrapper


class PyHugeClient:
    def __init__(
        self,
        url: str,
        graph: str,
        user: Optional[str] = None,
        pwd: Optional[str] = None,
        token: Optional[str] = None,
        graphspace: Optional[str] = None,
        timeout: Optional[tuple[float, float]] = None
    ):
        if token is None and (user is None or pwd is None):
            raise ValueError("Either a token or both username and password must be provided.")

        if token is not None and (user is not None or pwd is not None):
            log.warning(
                "Both token and username/password are provided. "
                "Token will be used for authentication."
            )

        self.cfg = HGraphConfig(
            url=url,
            graph_name=graph,
            token=token,
            username=user,
            password=pwd,
            graphspace=graphspace,
            timeout=timeout or (0.5, 15.0)
        )

    @manager_builder
    def schema(self) -> "SchemaManager":
        return SchemaManager

    @manager_builder
    def gremlin(self) -> "GremlinManager":
        return GremlinManager

    @manager_builder
    def graph(self) -> "GraphManager":
        return GraphManager

    @manager_builder
    def graphs(self) -> "GraphsManager":
        return GraphsManager

    @manager_builder
    def variable(self) -> "VariableManager":
        return VariableManager

    @manager_builder
    def auth(self) -> "AuthManager":
        return AuthManager

    @manager_builder
    def task(self) -> "TaskManager":
        return TaskManager

    @manager_builder
    def metrics(self) -> "MetricsManager":
        return MetricsManager

    @manager_builder
    def traverser(self) -> "TraverserManager":
        return TraverserManager

    @manager_builder
    def version(self) -> "VersionManager":
        return VersionManager

    def __repr__(self) -> str:
        return str(self.cfg)
