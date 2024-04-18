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
from typing import Optional, List, Dict, Any, Callable

import requests
from retry import retry

from hugegraph_llm.llms.base import BaseLLM
from hugegraph_llm.utils.config import Config
from hugegraph_llm.utils.constants import Constants


class ApiBotClient(BaseLLM):
    def __init__(self):
        self.c = Config(section=Constants.LLM_CONFIG)
        self.base_url = self.c.get_llm_url()

    @retry(tries=3, delay=1)
    def generate(
            self,
            messages: Optional[List[Dict[str, Any]]] = None,
            prompt: Optional[str] = None,
    ) -> str:
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]
        url = self.base_url

        payload = json.dumps({
            "messages": messages,
        })
        headers = {"Content-Type": "application/json"}
        response = requests.request("POST", url, headers=headers, data=payload, timeout=30)
        if response.status_code != 200:
            raise Exception(
                f"Request failed with code {response.status_code}, message: {response.text}"
            )
        response_json = json.loads(response.text)
        return response_json["content"]

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
        return 4096

    def get_llm_type(self) -> str:
        return "local_api"


if __name__ == "__main__":
    client = ApiBotClient()
    print(client.generate(prompt="What is the capital of China?"))
    print(client.generate(messages=[{"role": "user", "content": "What is the capital of China?"}]))
