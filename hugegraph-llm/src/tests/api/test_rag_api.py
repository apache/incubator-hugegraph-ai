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

import unittest
import asyncio
from fastapi import FastAPI, APIRouter
from fastapi.testclient import TestClient

from hugegraph_llm.api.rag_api import rag_http_api


class MockAsyncFunction:
    """Helper class to mock async functions"""

    def __init__(self, return_value):
        self.return_value = return_value
        self.called = False
        self.last_args = None
        self.last_kwargs = None

    async def __call__(self, *args, **kwargs):
        self.called = True
        self.last_args = args
        self.last_kwargs = kwargs
        return self.return_value


class TestRagApi(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.router = APIRouter()

        # Mock RAG answer function
        self.mock_rag_answer = MockAsyncFunction(
            ["Test raw answer", "Test vector answer", "Test graph answer", "Test combined answer"]
        )

        # Mock graph RAG recall function
        self.mock_graph_rag_recall = MockAsyncFunction({
            "query": "test query",
            "keywords": ["test", "keyword"],
            "match_vids": ["1", "2"],
            "graph_result_flag": True,
            "gremlin": "g.V().has('name', 'test')",
            "graph_result": ["result1", "result2"],
            "vertex_degree_list": [1, 2]
        })

        # Setup the API
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            rag_http_api(
                router=self.router,
                rag_answer_func=self.mock_rag_answer,
                graph_rag_recall_func=self.mock_graph_rag_recall
            )
        )

        self.app.include_router(self.router)
        self.client = TestClient(self.app)

    def test_rag_answer_api(self):
        """Test the /rag endpoint"""
        # Prepare test request
        request_data = {
            "query": "test query",
            "raw_answer": True,
            "vector_only": True,
            "graph_only": True,
            "graph_vector_answer": True
        }

        # Send request
        response = self.client.post("/rag", json=request_data)

        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.mock_rag_answer.called)
        self.assertEqual(self.mock_rag_answer.last_kwargs["text"], "test query")

        # Check response content
        response_data = response.json()
        self.assertEqual(response_data["query"], "test query")
        self.assertEqual(response_data["raw_answer"], "Test raw answer")
        self.assertEqual(response_data["vector_only"], "Test vector answer")
        self.assertEqual(response_data["graph_only"], "Test graph answer")
        self.assertEqual(response_data["graph_vector_answer"], "Test combined answer")

    def test_graph_rag_recall_api(self):
        """Test the /rag/graph endpoint"""
        # Prepare test request
        request_data = {
            "query": "test query",
            "gremlin_tmpl_num": 1,
            "rerank_method": "bleu",
            "near_neighbor_first": False,
            "custom_priority_info": "",
            "stream": False
        }

        # Send request
        response = self.client.post("/rag/graph", json=request_data)

        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.mock_graph_rag_recall.called)
        self.assertEqual(self.mock_graph_rag_recall.last_kwargs["query"], "test query")

        # Check response content
        response_data = response.json()
        self.assertIn("graph_recall", response_data)
        graph_recall = response_data["graph_recall"]
        self.assertEqual(graph_recall["query"], "test query")
        self.assertListEqual(graph_recall["keywords"], ["test", "keyword"])
        self.assertListEqual(graph_recall["match_vids"], ["1", "2"])
        self.assertTrue(graph_recall["graph_result_flag"])
        self.assertEqual(graph_recall["gremlin"], "g.V().has('name', 'test')")


if __name__ == "__main__":
    unittest.main()
