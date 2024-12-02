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


from hugegraph_llm.models.llms.ollama import OllamaClient
from hugegraph_llm.models.llms.openai import OpenAIClient
from hugegraph_llm.models.llms.qianfan import QianfanClient
from hugegraph_llm.config import settings


class LLMs:
    def __init__(self):
        self.chat_llm_type = settings.chat_llm_type
        self.extract_llm_type = settings.extract_llm_type
        self.text2gql_llm_type = settings.text2gql_llm_type

    def get_chat_llm(self):
        if self.chat_llm_type == "qianfan_wenxin":
            return QianfanClient(
                model_name=settings.qianfan_chat_language_model,
                api_key=settings.qianfan_chat_api_key,
                secret_key=settings.qianfan_chat_secret_key
            )
        if self.chat_llm_type == "openai":
            return OpenAIClient(
                api_key=settings.openai_chat_api_key,
                api_base=settings.openai_chat_api_base,
                model_name=settings.openai_chat_language_model,
                max_tokens=settings.openai_chat_tokens,
            )
        if self.chat_llm_type == "ollama/local":
            return OllamaClient(
                model=settings.ollama_chat_language_model,
                host=settings.ollama_chat_host,
                port=settings.ollama_chat_port,
            )
        raise Exception("chat llm type is not supported !")

    def get_extract_llm(self):
        if self.extract_llm_type == "qianfan_wenxin":
            return QianfanClient(
                model_name=settings.qianfan_extract_language_model,
                api_key=settings.qianfan_extract_api_key,
                secret_key=settings.qianfan_extract_secret_key
            )
        if self.extract_llm_type == "openai":
            return OpenAIClient(
                api_key=settings.openai_extract_api_key,
                api_base=settings.openai_extract_api_base,
                model_name=settings.openai_extract_language_model,
                max_tokens=settings.openai_extract_tokens,
            )
        if self.extract_llm_type == "ollama/local":
            return OllamaClient(
                model=settings.ollama_extract_language_model,
                host=settings.ollama_extract_host,
                port=settings.ollama_extract_port,
            )
        raise Exception("extract llm type is not supported !")

    def get_text2gql_llm(self):
        if self.text2gql_llm_type == "qianfan_wenxin":
            return QianfanClient(
                model_name=settings.qianfan_text2gql_language_model,
                api_key=settings.qianfan_text2gql_api_key,
                secret_key=settings.qianfan_text2gql_secret_key
            )
        if self.text2gql_llm_type == "openai":
            return OpenAIClient(
                api_key=settings.openai_text2gql_api_key,
                api_base=settings.openai_text2gql_api_base,
                model_name=settings.openai_text2gql_language_model,
                max_tokens=settings.openai_text2gql_tokens,
            )
        if self.text2gql_llm_type == "ollama/local":
            return OllamaClient(
                model=settings.ollama_text2gql_language_model,
                host=settings.ollama_text2gql_host,
                port=settings.ollama_text2gql_port,
            )
        raise Exception("text2gql llm type is not supported !")


if __name__ == "__main__":
    client = LLMs().get_chat_llm()
    print(client.generate(prompt="What is the capital of China?"))
    print(client.generate(messages=[{"role": "user", "content": "What is the capital of China?"}]))
