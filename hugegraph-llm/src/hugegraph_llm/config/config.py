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
from dataclasses import dataclass
from typing import Optional

import yaml
from dotenv import dotenv_values, set_key

from hugegraph_llm.config.config_data import ConfigData, PromptData
from hugegraph_llm.utils.log import log

dirname = os.path.dirname
package_path = dirname(dirname(dirname(dirname(os.path.abspath(__file__)))))
env_path = os.path.join(package_path, ".env")
yaml_file_path = os.path.join(package_path, "src/hugegraph_llm/resources/demo/config_prompt.yaml")


@dataclass
class Config(ConfigData):
    def from_env(self):
        if os.path.exists(env_path):
            env_config = read_dotenv()
            for key, value in env_config.items():
                if key in self.__annotations__ and value:
                    if self.__annotations__[key] in [int, Optional[int]]:
                        value = int(value)
                    setattr(self, key, value)
        else:
            self.generate_env()

    def generate_env(self):
        if os.path.exists(env_path):
            log.info("%s already exists, do you want to update it? (y/n)", env_path)
            update = input()
            if update.lower() != "y":
                return
            self.update_env()
        else:
            config_dict = {}
            for k, v in self.__dict__.items():
                config_dict[k] = v
            with open(env_path, "w", encoding="utf-8") as f:
                for k, v in config_dict.items():
                    if v is None:
                        f.write(f"{k}=\n")
                    else:
                        f.write(f"{k}={v}\n")
            log.info("Generate %s successfully!", env_path)

    def update_env(self):
        config_dict = {}
        for k, v in self.__dict__.items():
            config_dict[k] = str(v) if v else ""
        env_config = dotenv_values(f"{env_path}")
        for k, v in config_dict.items():
            if k in env_config and env_config[k] == v:
                continue
            log.info("Update %s: %s=%s", env_path, k, v)
            set_key(env_path, k, v, quote_mode="never")


def read_dotenv() -> dict[str, Optional[str]]:
    """Read a .env file in the given root path."""
    env_config = dotenv_values(f"{env_path}")
    log.info("Loading %s successfully!", env_path)
    for key, value in env_config.items():
        if key not in os.environ:
            os.environ[key] = value or ""
    return env_config


class PromptConfig(PromptData):

    def __init__(self):
        self.ensure_yaml_file_exists()

    def ensure_yaml_file_exists(self):
        if os.path.exists(yaml_file_path):
            log.info(f"Loading prompt file '{yaml_file_path}' successfully.")
            with open(yaml_file_path, "r") as file:
                data = yaml.safe_load(file)
                # Load existing values from the YAML file into the class attributes
                for key, value in data.items():
                    setattr(self, key, value)
        else:
            log.info(f"Prompt file '{yaml_file_path}' doesn't exist, create it.")
            self.save_to_yaml()

    def save_to_yaml(self):
        indented_schema = "\n".join([f"  {line}" for line in self.rag_schema.splitlines()])
        indented_example_prompt = "\n".join([f"    {line}" for line in self.schema_example_prompt.splitlines()])

        # This can be extended to add storage fields according to the data needs to be stored
        yaml_content = f"""{self.apache_license_header}

rag_schema: |
{indented_schema}

schema_example_prompt: |
{indented_example_prompt}

"""
        with open(yaml_file_path, "w") as file:
            file.write(yaml_content)
        log.info(f"YAML file '{yaml_file_path}' updated successfully.")

    def update_yaml_file(self):
        log.info(f"Updating '{yaml_file_path}' with the latest attributes.")
        self.save_to_yaml()
