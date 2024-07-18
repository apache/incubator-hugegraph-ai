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
from typing import Literal, Optional
from pathlib import Path
from dotenv import dotenv_values


@dataclass
class Config:
    """LLM settings"""
    # env_path: Optional[str] = ".env"
    llm_type: Literal["openai", "ollama", "qianfan_wenxin", "zhipu"] = "openai"
    embedding_type: Optional[Literal["openai", "ollama", "qianfan_wenxin", "zhipu"]] = "openai"
    # OpenAI settings
    openai_api_base: Optional[str] = "https://api.openai.com/v1"
    openai_api_key: Optional[str] = None
    openai_language_model: Optional[str] = "gpt-3.5-turbo"
    openai_embedding_model: Optional[str] = "text-embedding-3-small"
    openai_max_tokens: int = 4096
    # Ollama settings
    ollama_host: Optional[str] = "127.0.0.1"
    ollama_port: Optional[int] = 11434
    ollama_language_model: Optional[str] = None
    ollama_embedding_model: Optional[str] = None
    # QianFan/WenXin settings
    qianfan_api_key: Optional[str] = None
    qianfan_secret_key: Optional[str] = None
    qianfan_access_token: Optional[str] = None
    ## url settings
    qianfan_url_prefix: Optional[str] = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop"
    qianfan_chat_url: Optional[str] = qianfan_url_prefix + "/chat/"
    qianfan_language_model: Optional[str] = "ERNIE-4.0-Turbo-8K"
    qianfan_embed_url: Optional[str] = qianfan_url_prefix + "/embeddings/"
    # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu
    qianfan_embedding_model: Optional[str] = "embedding-v1"
    # Zhipu settings
    zhipu_api_key: Optional[str] = None
    zhipu_language_model: Optional[str] = "glm-4"
    zhipu_embedding_model: Optional[str] = "embedding-2"
    """HugeGraph settings"""
    graph_ip: Optional[str] = "127.0.0.1"
    graph_port: Optional[int] = 8080
    graph_space: Optional[str] = "DEFAULT"
    graph_name: Optional[str] = "hugegraph"
    graph_user: Optional[str] = "admin"
    graph_pwd: Optional[str] = "xxx"

    def from_env(self, config_dir: str):
        env_config = read_dotenv(config_dir)
        for key, value in env_config.items():
            if key in self.__annotations__ and value:
                if self.__annotations__[key] in [int, Optional[int]]:
                    value = int(value)
                setattr(self, key, value)

    def generate_env(self, config_dir: str):
        env_path = Path(config_dir) / ".env"
        config_dict = {}
        for k, v in self.__dict__.items():
            config_dict[k] = v
        with open(env_path, "w", encoding="utf-8") as f:
            for k, v in config_dict.items():
                if v is None:
                    f.write(f"{k}=\n")
                else:
                    f.write(f"{k}={v}\n")
        print(f"Generate {env_path} successfully!")


def read_dotenv(root: str) -> dict[str, Optional[str]]:
    """Read a .env file in the given root path."""
    env_path = Path(root) / ".env"
    if env_path.exists():
        env_config = dotenv_values(f"{env_path}")
        print(f"Read {env_path} successfully!")
        for key, value in env_config.items():
            if key not in os.environ:
                os.environ[key] = value or ""
        return env_config
    else:
        # TODO: generate a .env file
        pass
