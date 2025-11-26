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
import unittest

from hugegraph_llm.models.llms.ollama import OllamaClient


class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.skip_external = os.getenv("SKIP_EXTERNAL_SERVICES", "false").lower() == "true"

    @unittest.skipIf(os.getenv("SKIP_EXTERNAL_SERVICES", "false").lower() == "true",
                     "Skipping external service tests")
    def test_generate(self):
        ollama_client = OllamaClient(model="llama3:8b-instruct-fp16")
        response = ollama_client.generate(prompt="What is the capital of France?")
        print(response)

    @unittest.skipIf(os.getenv("SKIP_EXTERNAL_SERVICES", "false").lower() == "true",
                     "Skipping external service tests")
    def test_stream_generate(self):
        ollama_client = OllamaClient(model="llama3:8b-instruct-fp16")

        def on_token_callback(chunk):
            print(chunk, end="", flush=True)

        ollama_client.generate_streaming(
            prompt="What is the capital of France?", on_token_callback=on_token_callback
        )
