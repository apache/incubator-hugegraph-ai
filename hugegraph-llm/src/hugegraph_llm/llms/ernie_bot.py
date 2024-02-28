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


class ErnieBotClient(BaseLLM):
    def __init__(self):
        self.c = Config(section=Constants.LLM_CONFIG)
        self.api_key = self.c.get_llm_api_key()
        self.secret_key = self.c.get_llm_secret_key()
        self.base_url = self.c.get_llm_url()
        self.get_access_token()

    def get_access_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key,
        }
        return str(requests.post(url, params=params, timeout=2).json().get("access_token"))

    @retry(tries=3, delay=1)
    def generate(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
    ) -> str:
        if messages is None:
            assert prompt is not None, "Messages or prompt must be provided."
            messages = [{"role": "user", "content": prompt}]
        url = self.base_url + self.get_access_token()
        # parameter check failed, temperature range is (0, 1.0]
        payload = json.dumps({"messages": messages, "temperature": 0.1})
        headers = {"Content-Type": "application/json"}
        response = requests.request("POST", url, headers=headers, data=payload, timeout=30)
        if response.status_code != 200:
            raise Exception(
                f"Request failed with code {response.status_code}, message: {response.text}"
            )
        return json.loads(response.text)["result"]

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
        return 6000

    def get_llm_type(self) -> str:
        return "ernie"


if __name__ == "__main__":
    client = ErnieBotClient()
    print(client.generate(prompt="What is the capital of China?"))
    print(client.generate(messages=[{"role": "user", "content": "What is the capital of China?"}]))
