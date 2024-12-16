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
from typing import Optional, Literal

from .models import BaseConfig


class LLMConfig(BaseConfig):
    """LLM settings"""

    chat_llm_type: Literal["openai", "ollama/local", "qianfan_wenxin", "zhipu"] = "openai"
    extract_llm_type: Literal["openai", "ollama/local", "qianfan_wenxin", "zhipu"] = "openai"
    text2gql_llm_type: Literal["openai", "ollama/local", "qianfan_wenxin", "zhipu"] = "openai"
    embedding_type: Optional[Literal["openai", "ollama/local", "qianfan_wenxin", "zhipu"]] = "openai"
    reranker_type: Optional[Literal["cohere", "siliconflow"]] = None
    # 1. OpenAI settings
    openai_chat_api_base: Optional[str] = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_chat_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
    openai_chat_language_model: Optional[str] = "gpt-4o-mini"
    openai_extract_api_base: Optional[str] = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_extract_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
    openai_extract_language_model: Optional[str] = "gpt-4o-mini"
    openai_text2gql_api_base: Optional[str] = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_text2gql_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
    openai_text2gql_language_model: Optional[str] = "gpt-4o-mini"
    openai_embedding_api_base: Optional[str] = os.environ.get("OPENAI_EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    openai_embedding_api_key: Optional[str] = os.environ.get("OPENAI_EMBEDDING_API_KEY")
    openai_embedding_model: Optional[str] = "text-embedding-3-small"
    openai_chat_tokens: int = 8192
    openai_extract_tokens: int = 256
    openai_text2gql_tokens: int = 4096
    # 2. Rerank settings
    cohere_base_url: Optional[str] = os.environ.get("CO_API_URL", "https://api.cohere.com/v1/rerank")
    reranker_api_key: Optional[str] = None
    reranker_model: Optional[str] = None
    # 3. Ollama settings
    ollama_chat_host: Optional[str] = "127.0.0.1"
    ollama_chat_port: Optional[int] = 11434
    ollama_chat_language_model: Optional[str] = None
    ollama_extract_host: Optional[str] = "127.0.0.1"
    ollama_extract_port: Optional[int] = 11434
    ollama_extract_language_model: Optional[str] = None
    ollama_text2gql_host: Optional[str] = "127.0.0.1"
    ollama_text2gql_port: Optional[int] = 11434
    ollama_text2gql_language_model: Optional[str] = None
    ollama_embedding_host: Optional[str] = "127.0.0.1"
    ollama_embedding_port: Optional[int] = 11434
    ollama_embedding_model: Optional[str] = None
    # 4. QianFan/WenXin settings
    qianfan_chat_api_key: Optional[str] = None
    qianfan_chat_secret_key: Optional[str] = None
    qianfan_chat_access_token: Optional[str] = None
    qianfan_extract_api_key: Optional[str] = None
    qianfan_extract_secret_key: Optional[str] = None
    qianfan_extract_access_token: Optional[str] = None
    qianfan_text2gql_api_key: Optional[str] = None
    qianfan_text2gql_secret_key: Optional[str] = None
    qianfan_text2gql_access_token: Optional[str] = None
    qianfan_embedding_api_key: Optional[str] = None
    qianfan_embedding_secret_key: Optional[str] = None
    # 4.1 URL settings
    qianfan_url_prefix: Optional[str] = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop"
    qianfan_chat_url: Optional[str] = qianfan_url_prefix + "/chat/"
    qianfan_chat_language_model: Optional[str] = "ERNIE-Speed-128K"
    qianfan_extract_language_model: Optional[str] = "ERNIE-Speed-128K"
    qianfan_text2gql_language_model: Optional[str] = "ERNIE-Speed-128K"
    qianfan_embed_url: Optional[str] = qianfan_url_prefix + "/embeddings/"
    # refer https://cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu to get more details
    qianfan_embedding_model: Optional[str] = "embedding-v1"
    # TODO: To be confirmed, whether to configure
    # 5. ZhiPu(GLM) settings
    zhipu_chat_api_key: Optional[str] = None
    zhipu_chat_language_model: Optional[str] = "glm-4"
    zhipu_chat_embedding_model: Optional[str] = "embedding-2"
    zhipu_extract_api_key: Optional[str] = None
    zhipu_extract_language_model: Optional[str] = "glm-4"
    zhipu_extract_embedding_model: Optional[str] = "embedding-2"
    zhipu_text2gql_api_key: Optional[str] = None
    zhipu_text2gql_language_model: Optional[str] = "glm-4"
    zhipu_text2gql_embedding_model: Optional[str] = "embedding-2"
