# !/usr/bin/env python3
"""
file: base.py
author: wenyuxuan@baidu.com
"""

from pyvermeer.utils.log import log


class BaseModule:
    """基类"""

    def __init__(self, client):
        self._client = client
        self.log = log.getChild(__name__)

    @property
    def session(self):
        """返回客户端的session对象"""
        return self._client.session

    def _send_request(self, method: str, endpoint: str, params: dict = None):
        """统一请求入口"""
        self.log.debug(f"Sending {method} to {endpoint}")
        return self._client.send_request(
            method=method,
            endpoint=endpoint,
            params=params
        )
