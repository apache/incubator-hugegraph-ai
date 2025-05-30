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

from typing import Dict
from typing import Optional

from pyvermeer.api.base import BaseModule
from pyvermeer.api.graph import GraphModule
from pyvermeer.api.task import TaskModule
from pyvermeer.utils.log import log
from pyvermeer.utils.vermeer_config import VermeerConfig
from pyvermeer.utils.vermeer_requests import VermeerSession


class PyVermeerClient:
    """Vermeer API Client"""

    def __init__(
            self,
            ip: str,
            port: int,
            token: str,
            timeout: Optional[tuple[float, float]] = None,
            log_level: str = "INFO",
    ):
        """初始化客户端，包括配置和会话管理
        :param ip:
        :param port:
        :param token:
        :param timeout:
        :param log_level:
        """
        self.cfg = VermeerConfig(ip, port, token, timeout)
        self.session = VermeerSession(self.cfg)
        self._modules: Dict[str, BaseModule] = {
            "graph": GraphModule(self),
            "tasks": TaskModule(self)
        }
        log.setLevel(log_level)

    def __getattr__(self, name):
        """通过属性访问模块"""
        if name in self._modules:
            return self._modules[name]
        raise AttributeError(f"Module {name} not found")

    def send_request(self, method: str, endpoint: str, params: dict = None):
        """统一请求方法"""
        return self.session.request(method, endpoint, params)
