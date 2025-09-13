#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from PyCGraph import CStatus
from hugegraph_llm.config import LLMConfig
from hugegraph_llm.models.embeddings.litellm import LiteLLMEmbedding
from hugegraph_llm.models.embeddings.ollama import OllamaEmbedding
from hugegraph_llm.models.embeddings.openai import OpenAIEmbedding
from hugegraph_llm.models.llms.ollama import OllamaClient
from hugegraph_llm.models.llms.openai import OpenAIClient
from hugegraph_llm.models.llms.litellm import LiteLLMClient


def init_context(obj) -> CStatus:
    obj.context = obj.getGParamWithNoEmpty("wkflow_state")
    obj.wk_input = obj.getGParamWithNoEmpty("wkflow_input")
    return CStatus()


def get_embedding(llm_settings: LLMConfig):
    if llm_settings.embedding_type == "openai":
        return OpenAIEmbedding(
            model_name=llm_settings.openai_embedding_model,
            api_key=llm_settings.openai_embedding_api_key,
            api_base=llm_settings.openai_embedding_api_base,
        )
    if llm_settings.embedding_type == "ollama/local":
        return OllamaEmbedding(
            model_name=llm_settings.ollama_embedding_model,
            host=llm_settings.ollama_embedding_host,
            port=llm_settings.ollama_embedding_port,
        )
    if llm_settings.embedding_type == "litellm":
        return LiteLLMEmbedding(
            model_name=llm_settings.litellm_embedding_model,
            api_key=llm_settings.litellm_embedding_api_key,
            api_base=llm_settings.litellm_embedding_api_base,
        )

    raise Exception("embedding type is not supported !")


def get_chat_llm(llm_settings: LLMConfig):
    if llm_settings.chat_llm_type == "openai":
        return OpenAIClient(
            api_key=llm_settings.openai_chat_api_key,
            api_base=llm_settings.openai_chat_api_base,
            model_name=llm_settings.openai_chat_language_model,
            max_tokens=llm_settings.openai_chat_tokens,
        )
    if llm_settings.chat_llm_type == "ollama/local":
        return OllamaClient(
            model=llm_settings.ollama_chat_language_model,
            host=llm_settings.ollama_chat_host,
            port=llm_settings.ollama_chat_port,
        )
    if llm_settings.chat_llm_type == "litellm":
        return LiteLLMClient(
            api_key=llm_settings.litellm_chat_api_key,
            api_base=llm_settings.litellm_chat_api_base,
            model_name=llm_settings.litellm_chat_language_model,
            max_tokens=llm_settings.litellm_chat_tokens,
        )
    raise Exception("chat llm type is not supported !")


def get_extract_llm(llm_settings: LLMConfig):
    if llm_settings.extract_llm_type == "openai":
        return OpenAIClient(
            api_key=llm_settings.openai_extract_api_key,
            api_base=llm_settings.openai_extract_api_base,
            model_name=llm_settings.openai_extract_language_model,
            max_tokens=llm_settings.openai_extract_tokens,
        )
    if llm_settings.extract_llm_type == "ollama/local":
        return OllamaClient(
            model=llm_settings.ollama_extract_language_model,
            host=llm_settings.ollama_extract_host,
            port=llm_settings.ollama_extract_port,
        )
    if llm_settings.extract_llm_type == "litellm":
        return LiteLLMClient(
            api_key=llm_settings.litellm_extract_api_key,
            api_base=llm_settings.litellm_extract_api_base,
            model_name=llm_settings.litellm_extract_language_model,
            max_tokens=llm_settings.litellm_extract_tokens,
        )
    raise Exception("extract llm type is not supported !")


def get_text2gql_llm(llm_settings: LLMConfig):
    if llm_settings.text2gql_llm_type == "openai":
        return OpenAIClient(
            api_key=llm_settings.openai_text2gql_api_key,
            api_base=llm_settings.openai_text2gql_api_base,
            model_name=llm_settings.openai_text2gql_language_model,
            max_tokens=llm_settings.openai_text2gql_tokens,
        )
    if llm_settings.text2gql_llm_type == "ollama/local":
        return OllamaClient(
            model=llm_settings.ollama_text2gql_language_model,
            host=llm_settings.ollama_text2gql_host,
            port=llm_settings.ollama_text2gql_port,
        )
    if llm_settings.text2gql_llm_type == "litellm":
        return LiteLLMClient(
            api_key=llm_settings.litellm_text2gql_api_key,
            api_base=llm_settings.litellm_text2gql_api_base,
            model_name=llm_settings.litellm_text2gql_language_model,
            max_tokens=llm_settings.litellm_text2gql_tokens,
        )
    raise Exception("text2gql llm type is not supported !")
