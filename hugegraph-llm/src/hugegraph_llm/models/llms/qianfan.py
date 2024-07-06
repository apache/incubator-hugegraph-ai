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
from typing import Optional, List, Dict, Any, Callable

import qianfan
from retry import retry

from hugegraph_llm.models.llms.base import BaseLLM


class QianfanClient(BaseLLM):
    def __init__(self, model_name: Optional[str] = "ERNIE-4.0-Turbo-8K",
                 api_key: Optional[str] = None, secret_key: Optional[str] = None):
        qianfan.get_config().AK = api_key or os.getenv("QIANFAN_ACCESS_KEY")
        qianfan.get_config().SK = secret_key or os.getenv("QIANFAN_SECRET_KEY")
        self.chat_model = model_name
        self.chat_comp = qianfan.ChatCompletion()

    @retry(tries=3, delay=1)
    def generate(
            self,
            messages: Optional[List[Dict[str, Any]]] = None,
            prompt: Optional[str] = None,
    ) -> str:
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]

        response = self.chat_comp.do(model=self.chat_model, messages=messages)
        if response.code != 200:
            raise Exception(
                f"Request failed with code {response.code}, message: {response.body['error_msg']}"
            )
        return response.body["result"]

    def generate_streaming(
            self,
            messages: Optional[List[Dict[str, Any]]] = None,
            prompt: Optional[str] = None,
            on_token_callback: Callable = None,
    ) -> str:
        return self.generate(messages, prompt)

    def num_tokens_from_string(self, string: str) -> int:
        return len(string)

    def max_allowed_token_length(self) -> int:
        # TODO: replace with config way
        return 6000

    def get_llm_type(self) -> str:
        return "qianfan_wenxin"


if __name__ == "__main__":
    client = QianfanClient()
    print(client.generate(prompt="What is the capital of China?"))
    print(client.generate(messages=[{"role": "user", "content": "What is the capital of China?"}]))
