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

import configparser
import os


class Config:
    def __init__(self, config_file=None, section=None):
        if config_file is None:
            root_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            config_file = os.path.join(root_dir, "config", "config.ini")
        if section is None:
            raise Exception("config section cannot be none !")
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)
        self.section = section

    def get_config(self):
        return self.config

    def get_graph_ip(self):
        return self.config.get(self.section, "ip")

    def get_graph_port(self):
        return self.config.get(self.section, "port")

    def get_graph_user(self):
        return self.config.get(self.section, "user")

    def get_graph_pwd(self):
        return self.config.get(self.section, "pwd")

    def get_graph_name(self):
        return self.config.get(self.section, "graph")

    def get_llm_api_key(self):
        return self.config.get(self.section, "api_key")

    def get_llm_secret_key(self):
        return self.config.get(self.section, "secret_key")

    def get_llm_wenxin_url(self):
        return self.config.get(self.section, "wenxin_url")

    def get_llm_type(self):
        return self.config.get(self.section, "type")
