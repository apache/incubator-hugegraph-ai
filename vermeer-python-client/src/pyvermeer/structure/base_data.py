# !/usr/bin/env python3
"""
file: base_data.py
author: wenyuxuan@baidu.com
"""

RESPONSE_ERR = 1
RESPONSE_OK = 0
RESPONSE_NONE = -1


class BaseResponse(object):
    """
    Base response class
    """

    def __init__(self, dic: dict):
        """
        init
        :param dic:
        """
        self.__errcode = dic.get('errcode', RESPONSE_NONE)
        self.__message = dic.get('message', "")

    @property
    def errcode(self) -> int:
        """
        get error code
        :return:
        """
        return self.__errcode

    @property
    def message(self) -> str:
        """
        get message
        :return:
        """
        return self.__message
