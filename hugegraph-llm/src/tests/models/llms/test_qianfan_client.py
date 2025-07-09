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
from unittest.mock import patch, MagicMock, AsyncMock

try:
    from hugegraph_llm.models.llms.qianfan import QianfanClient
    QIANFAN_AVAILABLE = True
except ImportError:
    QIANFAN_AVAILABLE = False
    QianfanClient = None


@unittest.skipIf(not QIANFAN_AVAILABLE, "QianfanClient not available")
class TestQianfanClient(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with mocked qianfan configuration."""
        self.patcher = patch('hugegraph_llm.models.llms.qianfan.qianfan.get_config')
        self.mock_get_config = self.patcher.start()

        # Mock qianfan config
        mock_config = MagicMock()
        self.mock_get_config.return_value = mock_config

        # Mock ChatCompletion
        self.chat_comp_patcher = patch('hugegraph_llm.models.llms.qianfan.qianfan.ChatCompletion')
        self.mock_chat_completion_class = self.chat_comp_patcher.start()
        self.mock_chat_comp = MagicMock()
        self.mock_chat_completion_class.return_value = self.mock_chat_comp

    def tearDown(self):
        """Clean up patches."""
        self.patcher.stop()
        self.chat_comp_patcher.stop()

    def test_generate(self):
        """Test generate method with mocked response."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.code = 200
        mock_response.body = {
            "result": "Beijing is the capital of China.",
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
        }
        self.mock_chat_comp.do.return_value = mock_response

        # Test the method
        qianfan_client = QianfanClient()
        response = qianfan_client.generate(prompt="What is the capital of China?")

        # Verify the result
        self.assertIsInstance(response, str)
        self.assertEqual(response, "Beijing is the capital of China.")
        self.assertGreater(len(response), 0)

        # Verify the method was called with correct parameters
        self.mock_chat_comp.do.assert_called_once_with(
            model="ernie-4.5-8k-preview",
            messages=[{"role": "user", "content": "What is the capital of China?"}]
        )

    def test_generate_with_messages(self):
        """Test generate method with messages parameter."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.code = 200
        mock_response.body = {
            "result": "Beijing is the capital of China.",
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
        }
        self.mock_chat_comp.do.return_value = mock_response

        # Test the method
        qianfan_client = QianfanClient()
        messages = [{"role": "user", "content": "What is the capital of China?"}]
        response = qianfan_client.generate(messages=messages)

        # Verify the result
        self.assertIsInstance(response, str)
        self.assertEqual(response, "Beijing is the capital of China.")
        self.assertGreater(len(response), 0)

        # Verify the method was called with correct parameters
        self.mock_chat_comp.do.assert_called_once_with(
            model="ernie-4.5-8k-preview",
            messages=messages
        )

    def test_generate_error_response(self):
        """Test generate method with error response."""
        # Setup mock error response
        mock_response = MagicMock()
        mock_response.code = 400
        mock_response.body = {"error_msg": "Invalid request"}
        self.mock_chat_comp.do.return_value = mock_response

        # Test the method
        qianfan_client = QianfanClient()

        # Verify exception is raised
        with self.assertRaises(Exception) as cm:
            qianfan_client.generate(prompt="What is the capital of China?")

        self.assertIn("Request failed with code 400", str(cm.exception))
        self.assertIn("Invalid request", str(cm.exception))

    def test_agenerate(self):
        """Test agenerate method with mocked response."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.code = 200
        mock_response.body = {
            "result": "Beijing is the capital of China.",
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
        }

        # Use AsyncMock for async method
        self.mock_chat_comp.ado = AsyncMock(return_value=mock_response)

        qianfan_client = QianfanClient()

        async def run_async_test():
            response = await qianfan_client.agenerate(prompt="What is the capital of China?")
            self.assertIsInstance(response, str)
            self.assertEqual(response, "Beijing is the capital of China.")
            self.assertGreater(len(response), 0)

        asyncio.run(run_async_test())

        # Verify the method was called with correct parameters
        self.mock_chat_comp.ado.assert_called_once_with(
            model="ernie-4.5-8k-preview",
            messages=[{"role": "user", "content": "What is the capital of China?"}]
        )

    def test_agenerate_error_response(self):
        """Test agenerate method with error response."""
        # Setup mock error response
        mock_response = MagicMock()
        mock_response.code = 400
        mock_response.body = {"error_msg": "Invalid request"}

        # Use AsyncMock for async method
        self.mock_chat_comp.ado = AsyncMock(return_value=mock_response)

        qianfan_client = QianfanClient()

        async def run_async_test():
            with self.assertRaises(Exception) as cm:
                await qianfan_client.agenerate(prompt="What is the capital of China?")

            self.assertIn("Request failed with code 400", str(cm.exception))
            self.assertIn("Invalid request", str(cm.exception))

        asyncio.run(run_async_test())

    def test_generate_streaming(self):
        """Test generate_streaming method with mocked response."""
        # Setup mock streaming response
        mock_msgs = [
            MagicMock(body={"result": "Beijing "}),
            MagicMock(body={"result": "is the "}),
            MagicMock(body={"result": "capital of China."})
        ]
        self.mock_chat_comp.do.return_value = iter(mock_msgs)

        qianfan_client = QianfanClient()

        # Test callback function
        collected_tokens = []
        def on_token_callback(chunk):
            collected_tokens.append(chunk)

        # Test streaming generation
        response_generator = qianfan_client.generate_streaming(
            prompt="What is the capital of China?",
            on_token_callback=on_token_callback
        )

        # Collect all tokens
        tokens = list(response_generator)

        # Verify the results
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[0], "Beijing ")
        self.assertEqual(tokens[1], "is the ")
        self.assertEqual(tokens[2], "capital of China.")

        # Verify callback was called
        self.assertEqual(collected_tokens, tokens)

        # Verify the method was called with correct parameters
        self.mock_chat_comp.do.assert_called_once_with(
            messages=[{"role": "user", "content": "What is the capital of China?"}],
            model="ernie-4.5-8k-preview",
            stream=True
        )

    def test_num_tokens_from_string(self):
        """Test num_tokens_from_string method."""
        qianfan_client = QianfanClient()
        test_string = "Hello, world!"
        token_count = qianfan_client.num_tokens_from_string(test_string)
        self.assertEqual(token_count, len(test_string))

    def test_max_allowed_token_length(self):
        """Test max_allowed_token_length method."""
        qianfan_client = QianfanClient()
        max_tokens = qianfan_client.max_allowed_token_length()
        self.assertEqual(max_tokens, 6000)

    def test_get_llm_type(self):
        """Test get_llm_type method."""
        qianfan_client = QianfanClient()
        llm_type = qianfan_client.get_llm_type()
        self.assertEqual(llm_type, "qianfan_wenxin")
