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


class GraphInstance:
    def __init__(self, ip, port, graph_name, user_name, passwd, timeout):
        self.__ip = ip
        self.__port = port
        # self.__graphspace = graphspace
        self.__graph_name = graph_name
        self.__user_name = user_name
        self.__passwd = passwd
        self.__timeout = timeout

    @property
    def ip(self):
        return self.__ip

    @property
    def port(self):
        return self.__port

    @property
    def graph_name(self):
        return self.__graph_name

    @property
    def user_name(self):
        return self.__user_name

    @property
    def passwd(self):
        return self.__passwd

    @property
    def timeout(self):
        return self.__timeout

    def __repr__(self):
        res = (
            f"ip:{self.ip}, port:{self.port}, graph_name:{self.graph_name},"
            f" user_name:{self.user_name}, passwd:{self.passwd}, timeout:{self.timeout}"
        )
        return res
