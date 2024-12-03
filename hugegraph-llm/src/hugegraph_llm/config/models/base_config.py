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


import os

from dotenv import dotenv_values, set_key
from pydantic_settings import BaseSettings
from hugegraph_llm.utils.log import log

dir_name = os.path.dirname
package_path = dir_name(dir_name(dir_name(dir_name(dir_name(os.path.abspath(__file__))))))
env_path = os.path.join(package_path, ".env")



class BaseConfig(BaseSettings):
    class Config:
        env_file = env_path
        case_sensitive = False
        extra = 'ignore' # ignore extra fields to avoid ValidationError
        env_ignore_empty = True

    def generate_env(self):
        if os.path.exists(env_path):
            log.info("%s already exists, do you want to override with the default configuration? (y/n)", env_path)
            update = input()
            if update.lower() != "y":
                return
            self.update_env()
        else:
            config_dict = self.model_dump()
            config_dict = {k.upper(): v for k, v in config_dict.items()}
            with open(env_path, "w", encoding="utf-8") as f:
                for k, v in config_dict.items():
                    if v is None:
                        f.write(f"{k}=\n")
                    else:
                        f.write(f"{k}={v}\n")
            log.info("Generate %s successfully!", env_path)

    def update_env(self):
        config_dict = self.model_dump()
        config_dict = {k.upper(): v for k, v in config_dict.items()}
        env_config = dotenv_values(f"{env_path}")
        for k, v in config_dict.items():
            if k in env_config and env_config[k] == v:
                continue
            log.info("Update %s: %s=%s", env_path, k, v)
            set_key(env_path, k, v, quote_mode="never")

    def check_env(self):
        config_dict = self.model_dump()
        config_dict = {k.upper(): v for k, v in config_dict.items()}
        env_config = dotenv_values(f"{env_path}")
        for k, v in config_dict.items():
            if k in env_config:
                continue
            log.info("Update %s: %s=%s", env_path, k, v)
            set_key(env_path, k, v, quote_mode="never")

    def __init__(self, **data):
        super().__init__(**data)
        if not os.path.exists(env_path):
            self.generate_env()
        log.info("Loading %s successfully!", env_path)
