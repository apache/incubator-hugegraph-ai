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


import json

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Config:
    """LLM settings"""
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
    qianfan_url_prefix = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop"
    qianfan_chat_url: Optional[str] = qianfan_url_prefix + "/chat/"
    qianfan_language_model: Optional[str] = "completions_pro"
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
    graph_name: Optional[str] = "hugegraph"
    graph_user: Optional[str] = "admin"
    graph_pwd: Optional[str] = "xxx"

    def from_json(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        for k, v in d.items():
            setattr(self, k, v)
