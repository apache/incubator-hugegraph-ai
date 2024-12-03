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

import json
import re
from typing import Optional, List, Dict, Any, Union

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.utils.log import log


def gremlin_generate_prompt(
        query: str,
        schema: str,
        example: Optional[str] = None,
        vertices: Optional[List[str]] = None
) -> str:
    prompt = ""
    if example:
        prompt += f"""\
Given the example query-gremlin pairs:
{example}
"""
    prompt += f"""\
Given the graph schema:
```json
{schema}
```
"""
    if vertices:
        prompt += f"""\
Given the extracted vertex vid:
{vertices}
"""
    prompt += f"""\
Generate gremlin from the following user query.
{query}
The output format must be like:
```gremlin
g.V().limit(10)
```
The generated gremlin is:"""
    return prompt


class GremlinGenerateSynthesize:
    def __init__(
            self,
            llm: BaseLLM = None,
            schema: Optional[Union[dict, str]] = None,
            vertices: Optional[List[str]] = None
    ) -> None:
        self.llm = llm or LLMs().get_text2gql_llm()
        if isinstance(schema, dict):
            schema = json.dumps(schema, ensure_ascii=False)
        self.schema = schema
        self.vertices = vertices

    def _extract_gremlin(self, response: str) -> str:
        match = re.search("```gremlin.*```", response, re.DOTALL)
        assert match is not None, f"No gremlin found in response: {response}"
        return match.group()[len("```gremlin"):-len("```")].strip()

    def _format_examples(self, examples: Optional[List[Dict[str, str]]]) -> Optional[str]:
        if not examples:
            return None
        example_strings = []
        for example in examples:
            example_strings.append(
                f"- query: {example['query']}\n"
                f"- gremlin:\n```gremlin\n{example['gremlin']}\n```")
        return "\n\n".join(example_strings)

    def _format_vertices(self, vertices: Optional[List[str]]) -> Optional[str]:
        if not vertices:
            return None
        return "\n".join([f"- {vid}" for vid in vertices])

    async def async_generate(self, context: Dict[str, Any]):
        async_tasks = {}
        query = context.get("query")
        raw_example = [{'query':'who is peter', 'gremlin':"g.V().has('name', 'peter')"}]
        raw_prompt = gremlin_generate_prompt(
            query,
            self.schema,
            self._format_examples(examples=raw_example),
            self._format_vertices(vertices=self.vertices)
        )
        async_tasks["raw_answer"] = asyncio.create_task(self.llm.agenerate(prompt = raw_prompt))

        examples = context.get("match_result")
        prompt = gremlin_generate_prompt(
            query,
            self.schema,
            self._format_examples(examples=examples),
            self._format_vertices(vertices=self.vertices)
        )
        async_tasks["initilized_answer"] = asyncio.create_task(self.llm.agenerate(prompt = prompt))
        
        raw_response = await async_tasks["raw_answer"]
        initialized_response = await async_tasks["initilized_answer"]

        context["result"] = self._extract_gremlin(response=initialized_response)
        context["raw_result"] = self._extract_gremlin(response=raw_response)
        context["call_count"] = context.get("call_count", 0) + 2


        return context

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        if not query:
                raise ValueError("query is required")

        context = asyncio.run(self.async_generate(context))
        return context
