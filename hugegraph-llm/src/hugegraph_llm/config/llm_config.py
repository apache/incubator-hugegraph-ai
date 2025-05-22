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
from typing import Literal, Optional

from .models import BaseConfig


class LLMConfig(BaseConfig):
    """LLM settings"""

    chat_llm_type: Literal["openai", "litellm", "ollama/local", "qianfan_wenxin"] = "openai"
    extract_llm_type: Literal["openai", "litellm", "ollama/local", "qianfan_wenxin"] = "openai"
    text2gql_llm_type: Literal["openai", "litellm", "ollama/local", "qianfan_wenxin"] = "openai"
    embedding_type: Optional[Literal["openai", "litellm", "ollama/local", "qianfan_wenxin"]] = "openai"
    reranker_type: Optional[Literal["cohere", "siliconflow"]] = None
    # 1. OpenAI settings
    openai_chat_api_base: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_chat_api_key: str | None = os.environ.get("OPENAI_API_KEY")
    openai_chat_language_model: str = "gpt-4.1-mini"
    openai_extract_api_base: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_extract_api_key: str | None = os.environ.get("OPENAI_API_KEY")
    openai_extract_language_model: str = "gpt-4.1-mini"
    openai_text2gql_api_base: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_text2gql_api_key: str | None = os.environ.get("OPENAI_API_KEY")
    openai_text2gql_language_model: str = "gpt-4.1-mini"
    openai_embedding_api_base: str = os.environ.get("OPENAI_EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    openai_embedding_api_key: str | None = os.environ.get("OPENAI_EMBEDDING_API_KEY")
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_model_dim: int = 1536
    openai_chat_tokens: int = 8192
    openai_extract_tokens: int = 256
    openai_text2gql_tokens: int = 4096
    # 2. Rerank settings
    cohere_base_url: str = os.environ.get("CO_API_URL", "https://api.cohere.com/v1/rerank")
    reranker_api_key: Optional[str] = None
    reranker_model: Optional[str] = None
    # 3. Ollama settings
    ollama_chat_host: str = "127.0.0.1"
    ollama_chat_port: int = 11434
    ollama_chat_language_model: str | None = None
    ollama_extract_host: str = "127.0.0.1"
    ollama_extract_port: int = 11434
    ollama_extract_language_model: str | None = None
    ollama_text2gql_host: str = "127.0.0.1"
    ollama_text2gql_port: int = 11434
    ollama_text2gql_language_model: str | None = None
    ollama_embedding_host: str = "127.0.0.1"
    ollama_embedding_port: int = int(11434)
    ollama_embedding_model: str = 'quentinz/bge-large-zh-v1.5'
    ollama_embedding_model_dim: Optional[int] = (
        int(os.getenv("OLLAMA_EMBEDDING_MODEL_DIM")) if os.getenv("OLLAMA_EMBEDDING_MODEL_DIM") else None  # type:ignore
    )

    # 4. QianFan/WenXin settings
    # TODO: update to one token key mode
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
    qianfan_url_prefix: str = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop"
    qianfan_chat_url: str = qianfan_url_prefix + "/chat/"
    qianfan_chat_language_model: str = "ERNIE-Speed-128K"
    qianfan_extract_language_model: str = "ERNIE-Speed-128K"
    qianfan_text2gql_language_model: str = "ERNIE-Speed-128K"
    qianfan_embed_url: str = qianfan_url_prefix + "/embeddings/"
    qianfan_embedding_model_dim: int = 384

    # refer https://cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu to get more details
    qianfan_embedding_model: str = "embedding-v1"
    # 5. LiteLLM settings
    litellm_chat_api_key: Optional[str] = None
    litellm_chat_api_base: Optional[str] = None
    litellm_chat_language_model: str = "openai/gpt-4.1-mini"
    litellm_chat_tokens: int = 8192
    litellm_extract_api_key: Optional[str] = None
    litellm_extract_api_base: Optional[str] = None
    litellm_extract_language_model: str = "openai/gpt-4.1-mini"
    litellm_extract_tokens: int = 256
    litellm_text2gql_api_key: Optional[str] = None
    litellm_text2gql_api_base: Optional[str] = None
    litellm_text2gql_language_model: str = "openai/gpt-4.1-mini"
    litellm_text2gql_tokens: int = 4096
    litellm_embedding_api_key: Optional[str] = None
    litellm_embedding_api_base: Optional[str] = None
    litellm_embedding_model: str = "openai/text-embedding-3-small"
    litellm_embedding_model_dim: int = 1536
