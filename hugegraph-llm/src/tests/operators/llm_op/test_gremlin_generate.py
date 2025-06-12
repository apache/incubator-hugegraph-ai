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

# pylint: disable=protected-access,no-member

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.operators.llm_op.gremlin_generate import GremlinGenerateSynthesize


class TestGremlinGenerateSynthesize(unittest.TestCase):
    def setUp(self):
        # Create mock LLM
        self.mock_llm = MagicMock(spec=BaseLLM)
        self.mock_llm.agenerate = AsyncMock()

        # Sample schema
        self.schema = {
            "vertexLabels": [
                {"name": "person", "properties": ["name", "age"]},
                {"name": "movie", "properties": ["title", "year"]},
            ],
            "edgeLabels": [{"name": "acted_in", "sourceLabel": "person", "targetLabel": "movie"}],
        }

        # Sample vertices
        self.vertices = ["person:1", "movie:2"]

        # Sample query
        self.query = "Find all movies that Tom Hanks acted in"

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with patch("hugegraph_llm.operators.llm_op.gremlin_generate.LLMs") as mock_llms_class:
            mock_llms_instance = MagicMock()
            mock_llms_instance.get_text2gql_llm.return_value = self.mock_llm
            mock_llms_class.return_value = mock_llms_instance

            generator = GremlinGenerateSynthesize()

            self.assertEqual(generator.llm, self.mock_llm)
            self.assertIsNone(generator.schema)
            self.assertIsNone(generator.vertices)
            self.assertIsNotNone(generator.gremlin_prompt)

    def test_init_with_parameters(self):
        """Test initialization with provided parameters."""
        custom_prompt = "Custom prompt template: {query}, {schema}, {example}, {vertices}"

        generator = GremlinGenerateSynthesize(
            llm=self.mock_llm,
            schema=self.schema,
            vertices=self.vertices,
            gremlin_prompt=custom_prompt,
        )

        self.assertEqual(generator.llm, self.mock_llm)
        self.assertEqual(generator.schema, json.dumps(self.schema, ensure_ascii=False))
        self.assertEqual(generator.vertices, self.vertices)
        self.assertEqual(generator.gremlin_prompt, custom_prompt)

    def test_init_with_string_schema(self):
        """Test initialization with schema as string."""
        schema_str = json.dumps(self.schema, ensure_ascii=False)

        generator = GremlinGenerateSynthesize(llm=self.mock_llm, schema=schema_str)

        self.assertEqual(generator.schema, schema_str)

    def test_extract_gremlin(self):
        """Test the _extract_gremlin method."""
        generator = GremlinGenerateSynthesize(llm=self.mock_llm)

        # Test with valid gremlin code block
        response = (
            "Here is the Gremlin query:\n```gremlin\n"
            "g.V().has('person', 'name', 'Tom Hanks').out('acted_in')\n```"
        )
        gremlin = generator._extract_gremlin(response)
        self.assertEqual(gremlin, "g.V().has('person', 'name', 'Tom Hanks').out('acted_in')")

        # Test with invalid response
        with self.assertRaises(AssertionError):
            generator._extract_gremlin("No gremlin code block here")

    def test_format_examples(self):
        """Test the _format_examples method."""
        generator = GremlinGenerateSynthesize(llm=self.mock_llm)

        # Test with valid examples
        examples = [
            {"query": "who is Tom Hanks", "gremlin": "g.V().has('person', 'name', 'Tom Hanks')"},
            {
                "query": "what movies did Tom Hanks act in",
                "gremlin": "g.V().has('person', 'name', 'Tom Hanks').out('acted_in')",
            },
        ]

        formatted = generator._format_examples(examples)
        self.assertIn("who is Tom Hanks", formatted)
        self.assertIn("g.V().has('person', 'name', 'Tom Hanks')", formatted)
        self.assertIn("what movies did Tom Hanks act in", formatted)

        # Test with empty examples
        self.assertIsNone(generator._format_examples([]))
        self.assertIsNone(generator._format_examples(None))

    def test_format_vertices(self):
        """Test the _format_vertices method."""
        generator = GremlinGenerateSynthesize(llm=self.mock_llm)

        # Test with valid vertices
        vertices = ["person:1", "movie:2", "person:3"]
        formatted = generator._format_vertices(vertices)
        self.assertIn("- 'person:1'", formatted)
        self.assertIn("- 'movie:2'", formatted)
        self.assertIn("- 'person:3'", formatted)

        # Test with empty vertices
        self.assertIsNone(generator._format_vertices([]))
        self.assertIsNone(generator._format_vertices(None))

    @patch("asyncio.run")
    def test_run_with_valid_query(self, mock_asyncio_run):
        """Test the run method with a valid query."""
        # Setup mock for async_generate
        mock_context = {
            "query": self.query,
            "result": "g.V().has('person', 'name', 'Tom Hanks').out('acted_in')",
            "raw_result": "g.V().has('person', 'name', 'Tom Hanks').out('acted_in')",
            "call_count": 2,
        }
        mock_asyncio_run.return_value = mock_context

        # Create generator and run
        generator = GremlinGenerateSynthesize(llm=self.mock_llm)
        result = generator.run({"query": self.query})

        # Verify results
        mock_asyncio_run.assert_called_once()
        self.assertEqual(result["query"], self.query)
        self.assertEqual(
            result["result"], "g.V().has('person', 'name', 'Tom Hanks').out('acted_in')"
        )
        self.assertEqual(
            result["raw_result"], "g.V().has('person', 'name', 'Tom Hanks').out('acted_in')"
        )
        self.assertEqual(result["call_count"], 2)

    def test_run_with_empty_query(self):
        """Test the run method with an empty query."""
        generator = GremlinGenerateSynthesize(llm=self.mock_llm)

        with self.assertRaises(ValueError):
            generator.run({})

        with self.assertRaises(ValueError):
            generator.run({"query": ""})

    @patch("asyncio.create_task")
    @patch("asyncio.run")
    def test_async_generate(self, mock_asyncio_run, mock_create_task):
        """Test the async_generate method."""
        # Setup mocks for async tasks
        mock_raw_task = MagicMock()
        mock_raw_task.__await__ = lambda _: iter([None])
        mock_raw_task.return_value = "```gremlin\ng.V().has('person', 'name', 'Tom Hanks')\n```"

        mock_init_task = MagicMock()
        mock_init_task.__await__ = lambda _: iter([None])
        mock_init_task.return_value = (
            "```gremlin\ng.V().has('person', 'name', 'Tom Hanks').out('acted_in')\n```"
        )

        mock_create_task.side_effect = [mock_raw_task, mock_init_task]

        # Create generator and context
        generator = GremlinGenerateSynthesize(
            llm=self.mock_llm, schema=self.schema, vertices=self.vertices
        )

        # Mock asyncio.run to simulate running the coroutine
        mock_context = {
            "query": self.query,
            "result": "g.V().has('person', 'name', 'Tom Hanks').out('acted_in')",
            "raw_result": "g.V().has('person', 'name', 'Tom Hanks')",
            "call_count": 2,
        }
        mock_asyncio_run.return_value = mock_context

        # Run the method through run which uses asyncio.run
        result = generator.run({"query": self.query})

        # Verify results
        self.assertEqual(result["query"], self.query)
        self.assertEqual(
            result["result"], "g.V().has('person', 'name', 'Tom Hanks').out('acted_in')"
        )
        self.assertEqual(result["raw_result"], "g.V().has('person', 'name', 'Tom Hanks')")
        self.assertEqual(result["call_count"], 2)


if __name__ == "__main__":
    unittest.main()
