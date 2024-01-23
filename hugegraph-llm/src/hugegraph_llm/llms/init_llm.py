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

from hugegraph_llm.llms.openai import OpenAIChat
from hugegraph_llm.llms.ernie_bot import ErnieBotClient
from hugegraph_llm.utils.config import Config
from hugegraph_llm.utils.constants import Constants


class LLMs:
    def __init__(self):
        self.config = Config(section=Constants.LLM_CONFIG)
        self.config.get_llm_type()

    def get_llm(self):
        if self.config.get_llm_type() == "ernie":
            return ErnieBotClient()
        if self.config.get_llm_type() == "openai":
            return OpenAIChat(
                api_key=self.config.get_llm_api_key(),
                model_name=self.config.get_llm_model_name(),
                max_tokens=self.config.get_llm_max_token(),
            )
        raise Exception("llm type is not supported !")


if __name__ == "__main__":
    client = LLMs().get_llm()
    print(client.generate(prompt="What is the capital of China?"))
    print(client.generate(messages=[{"role": "user", "content": "What is the capital of China?"}]))
