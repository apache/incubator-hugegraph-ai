# !/usr/bin/env python3
"""
file: client.py
author: wenyuxuan@baidu.com
"""

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
