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


from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Config:
    """Model settings"""
    llm_type: Literal["openai", "ollama", "ernie", "zhipu"] = "openai"
    embedding_type: Optional[Literal["openai", "ollama", "ernie", "zhipu"]] = "openai"
    openai_api_base: Optional[str] = "https://api.openai.com/v1"
    openai_api_key: Optional[str] = None
    openai_language_model: Optional[str] = "gpt-3.5-turbo"
    openai_embedding_model: Optional[str] = "text-embedding-3-small"
    openai_max_tokens: int = 4096
    ollama_host: Optional[str] = "127.0.0.1"
    ollama_port: Optional[int] = 11434
    ollama_language_model: Optional[str] = None
    ollama_embedding_model: Optional[str] = None
    ernie_url: Optional[str] = ("https://aip.baidubce.com/rpc/2.0/ai_custom/"
                                "v1/wenxinworkshop/chat/completions_pro?access_token=")
    ernie_api_key: Optional[str] = None
    ernie_secret_key: Optional[str] = None
    ernie_model_name: Optional[str] = "wenxin"
    zhipu_api_key: Optional[str] = None
    zhipu_language_model: Optional[str] = "glm-4"
    zhipu_embedding_model: Optional[str] = "embedding-2"
    """HugeGraph settings"""
    graph_ip: Optional[str] = "127.0.0.1"
    graph_port: Optional[int] = 8080
    graph_name: Optional[str] = "hugegraph"
    graph_user: Optional[str] = "admin"
    graph_pwd: Optional[str] = "admin"
