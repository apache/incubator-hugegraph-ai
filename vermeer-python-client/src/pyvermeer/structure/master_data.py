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

import datetime

from pyvermeer.structure.base_data import BaseResponse
from pyvermeer.utils.vermeer_datetime import parse_vermeer_time


class MasterInfo:
    """Master 信息"""

    def __init__(self, dic: dict):
        """初始化函数"""
        self.__grpc_peer = dic.get('grpc_peer', '')
        self.__ip_addr = dic.get('ip_addr', '')
        self.__debug_mod = dic.get('debug_mod', False)
        self.__version = dic.get('version', '')
        self.__launch_time = parse_vermeer_time(dic.get('launch_time', ''))

    @property
    def grpc_peer(self) -> str:
        """gRPC地址"""
        return self.__grpc_peer

    @property
    def ip_addr(self) -> str:
        """IP地址"""
        return self.__ip_addr

    @property
    def debug_mod(self) -> bool:
        """是否为调试模式"""
        return self.__debug_mod

    @property
    def version(self) -> str:
        """Master 版本号"""
        return self.__version

    @property
    def launch_time(self) -> datetime:
        """Master 启动时间"""
        return self.__launch_time

    def to_dict(self):
        """返回字典格式数据"""
        return {
            "grpc_peer": self.__grpc_peer,
            "ip_addr": self.__ip_addr,
            "debug_mod": self.__debug_mod,
            "version": self.__version,
            "launch_time": self.__launch_time.strftime("%Y-%m-%d %H:%M:%S") if self.__launch_time else ''
        }


class MasterResponse(BaseResponse):
    """Master 响应"""

    def __init__(self, dic: dict):
        """初始化函数"""
        super().__init__(dic)
        self.__master_info = MasterInfo(dic['master_info'])

    @property
    def master_info(self) -> MasterInfo:
        """获取主节点信息"""
        return self.__master_info
