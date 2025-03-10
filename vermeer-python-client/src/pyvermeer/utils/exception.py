# !/usr/bin/env python3
"""
file: exceptions.py
author: wenyuxuan@baidu.com
"""


class ConnectError(Exception):
    """Raised when there is an issue connecting to the server."""

    def __init__(self, message):
        super().__init__(f"Connection error: {str(message)}")


class TimeOutError(Exception):
    """Raised when a request times out."""

    def __init__(self, message):
        super().__init__(f"Request timed out: {str(message)}")


class JsonDecodeError(Exception):
    """Raised when the response from the server cannot be decoded as JSON."""

    def __init__(self, message):
        super().__init__(f"Failed to decode JSON response: {str(message)}")


class UnknownError(Exception):
    """Raised for any other unknown errors."""

    def __init__(self, message):
        super().__init__(f"Unknown API error: {str(message)}")
