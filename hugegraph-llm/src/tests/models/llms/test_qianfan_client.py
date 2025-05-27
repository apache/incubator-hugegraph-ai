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

from hugegraph_llm.models.llms.qianfan import QianfanClient


class TestQianfanClient(unittest.TestCase):
    def test_generate(self):
        qianfan_client = QianfanClient()
        response = qianfan_client.generate(prompt="What is the capital of China?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_generate_with_messages(self):
        qianfan_client = QianfanClient()
        messages = [{"role": "user", "content": "What is the capital of China?"}]
        response = qianfan_client.generate(messages=messages)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_agenerate(self):
        qianfan_client = QianfanClient()

        async def run_async_test():
            response = await qianfan_client.agenerate(prompt="What is the capital of China?")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)

        asyncio.run(run_async_test())

    def test_generate_streaming(self):
        qianfan_client = QianfanClient()

        def on_token_callback(chunk):
            # This is a no-op in Qianfan's implementation
            pass

        response = qianfan_client.generate_streaming(
            prompt="What is the capital of China?", on_token_callback=on_token_callback
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_num_tokens_from_string(self):
        qianfan_client = QianfanClient()
        test_string = "Hello, world!"
        token_count = qianfan_client.num_tokens_from_string(test_string)
        self.assertEqual(token_count, len(test_string))

    def test_max_allowed_token_length(self):
        qianfan_client = QianfanClient()
        max_tokens = qianfan_client.max_allowed_token_length()
        self.assertEqual(max_tokens, 6000)

    def test_get_llm_type(self):
        qianfan_client = QianfanClient()
        llm_type = qianfan_client.get_llm_type()
        self.assertEqual(llm_type, "qianfan_wenxin")
