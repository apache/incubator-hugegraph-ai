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
        self.llm_type = settings.llm_type

    def get_llm(self):
        if self.llm_type == "qianfan_wenxin":
            return QianfanClient(
                model_name=settings.qianfan_language_model,
                api_key=settings.qianfan_api_key,
                secret_key=settings.qianfan_secret_key
            )
        if self.llm_type == "openai":
            return OpenAIClient(
                api_key=settings.openai_api_key,
                api_base=settings.openai_api_base,
                model_name=settings.openai_language_model,
                max_tokens=settings.openai_max_tokens,
            )
        if self.llm_type == "ollama":
            return OllamaClient(model=settings.ollama_language_model)
        raise Exception("llm type is not supported !")


if __name__ == "__main__":
    client = LLMs().get_llm()
    print(client.generate(prompt="What is the capital of China?"))
    print(client.generate(messages=[{"role": "user", "content": "What is the capital of China?"}]))
