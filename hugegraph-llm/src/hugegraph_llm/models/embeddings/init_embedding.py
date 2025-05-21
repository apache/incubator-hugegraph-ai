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
from hugegraph_llm.models.embeddings.litellm import LiteLLMEmbedding
from hugegraph_llm.models.embeddings.ollama import OllamaEmbedding
from hugegraph_llm.models.embeddings.openai import OpenAIEmbedding
from hugegraph_llm.models.embeddings.qianfan import QianFanEmbedding


class Embeddings:
    def __init__(self):
        self.embedding_type = llm_settings.embedding_type

    def get_embedding(self):
        if self.embedding_type == "openai":
            assert llm_settings.openai_embedding_model_dim, 'openai_embedding_model_dim is need'
            return OpenAIEmbedding(
                embedding_dimension=llm_settings.openai_embedding_model_dim,
                model_name=llm_settings.openai_embedding_model,
                api_key=llm_settings.openai_embedding_api_key,
                api_base=llm_settings.openai_embedding_api_base,
            )
        if self.embedding_type == "ollama/local":
            assert llm_settings.ollama_embedding_model_dim, 'ollama_embedding_model_dim is need'
            return OllamaEmbedding(
                embedding_dimension=llm_settings.ollama_embedding_model_dim,
                model=llm_settings.ollama_embedding_model,
                host=llm_settings.ollama_embedding_host,
                port=llm_settings.ollama_embedding_port,
            )
        if self.embedding_type == "qianfan_wenxin":
            return QianFanEmbedding(
                embedding_dimension=llm_settings.litellm_embedding_model_dim,
                model_name=llm_settings.qianfan_embedding_model,
                api_key=llm_settings.qianfan_embedding_api_key,
                secret_key=llm_settings.qianfan_embedding_secret_key,
            )  # type: ignore
        if self.embedding_type == "litellm":
            return LiteLLMEmbedding(
                embedding_dimension=llm_settings.litellm_embedding_model_dim,
                model_name=llm_settings.litellm_embedding_model,
                api_key=llm_settings.litellm_embedding_api_key,
                api_base=llm_settings.litellm_embedding_api_base,
            )  # type: ignore

        raise Exception("embedding type is not supported !")
