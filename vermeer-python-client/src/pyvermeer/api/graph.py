# !/usr/bin/env python3
"""
file: graph.py
author: wenyuxuan@baidu.com
"""

from pyvermeer.structure.graph_data import GraphsResponse, GraphResponse
from .base import BaseModule


class GraphModule(BaseModule):
    """Graph"""

    def get_graph(self, graph_name: str) -> GraphResponse:
        """获取任务列表"""
        response = self._send_request(
            "GET",
            f"/graphs/{graph_name}"
        )
        return GraphResponse(response)

    def get_graphs(self) -> GraphsResponse:
        """获取任务列表"""
        response = self._send_request(
            "GET",
            "/graphs",
        )
        return GraphsResponse(response)
