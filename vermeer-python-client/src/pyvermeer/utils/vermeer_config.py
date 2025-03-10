# !/usr/bin/env python3
"""
file: vermeer_config.py
author: wenyuxuan@baidu.com
"""


class VermeerConfig:
    """The configuration of a Vermeer instance."""
    ip: str
    port: int
    token: str
    factor: str
    username: str
    graph_space: str

    def __init__(self,
                 ip: str,
                 port: int,
                 token: str,
                 timeout: tuple[float, float] = (0.5, 15.0)):
        """Initialize the configuration for a Vermeer instance."""
        self.ip = ip
        self.port = port
        self.token = token
        self.timeout = timeout
