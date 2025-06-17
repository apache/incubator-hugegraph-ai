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

import asyncio
import unittest

from hugegraph_llm.models.llms.openai import OpenAIClient


class TestOpenAIClient(unittest.TestCase):
    def test_generate(self):
        openai_client = OpenAIClient(model_name="gpt-3.5-turbo")
        response = openai_client.generate(prompt="What is the capital of France?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_generate_with_messages(self):
        openai_client = OpenAIClient(model_name="gpt-3.5-turbo")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        response = openai_client.generate(messages=messages)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_agenerate(self):
        openai_client = OpenAIClient(model_name="gpt-3.5-turbo")

        async def run_async_test():
            response = await openai_client.agenerate(prompt="What is the capital of France?")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)

        asyncio.run(run_async_test())

    def test_stream_generate(self):
        openai_client = OpenAIClient(model_name="gpt-3.5-turbo")
        collected_tokens = []

        def on_token_callback(chunk):
            collected_tokens.append(chunk)

        response = openai_client.generate_streaming(
            prompt="What is the capital of France?", on_token_callback=on_token_callback
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertGreater(len(collected_tokens), 0)

    def test_num_tokens_from_string(self):
        openai_client = OpenAIClient(model_name="gpt-3.5-turbo")
        token_count = openai_client.num_tokens_from_string("Hello, world!")
        self.assertIsInstance(token_count, int)
        self.assertGreater(token_count, 0)

    def test_max_allowed_token_length(self):
        openai_client = OpenAIClient(model_name="gpt-3.5-turbo")
        max_tokens = openai_client.max_allowed_token_length()
        self.assertIsInstance(max_tokens, int)
        self.assertGreater(max_tokens, 0)

    def test_get_llm_type(self):
        openai_client = OpenAIClient()
        llm_type = openai_client.get_llm_type()
        self.assertEqual(llm_type, "openai")
