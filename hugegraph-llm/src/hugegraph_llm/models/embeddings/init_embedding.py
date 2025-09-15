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


from hugegraph_llm.config import llm_settings
from hugegraph_llm.config import LLMConfig
from hugegraph_llm.models.embeddings.litellm import LiteLLMEmbedding
from hugegraph_llm.models.embeddings.ollama import OllamaEmbedding
from hugegraph_llm.models.embeddings.openai import OpenAIEmbedding

model_map = {
    "openai": llm_settings.openai_embedding_model,
    "ollama/local": llm_settings.ollama_embedding_model,
    "litellm": llm_settings.litellm_embedding_model,
}


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


class Embeddings:
    def __init__(self):
        self.embedding_type = llm_settings.embedding_type

    def get_embedding(self):
        if self.embedding_type == "openai":
            return OpenAIEmbedding(
                model_name=llm_settings.openai_embedding_model,
                api_key=llm_settings.openai_embedding_api_key,
                api_base=llm_settings.openai_embedding_api_base,
            )
        if self.embedding_type == "ollama/local":
            return OllamaEmbedding(
                model_name=llm_settings.ollama_embedding_model,
                host=llm_settings.ollama_embedding_host,
                port=llm_settings.ollama_embedding_port,
            )
        if self.embedding_type == "litellm":
            return LiteLLMEmbedding(
                model_name=llm_settings.litellm_embedding_model,
                api_key=llm_settings.litellm_embedding_api_key,
                api_base=llm_settings.litellm_embedding_api_base,
            )

        raise Exception("embedding type is not supported !")
