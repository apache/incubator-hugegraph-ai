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


import re
import json
from typing import Optional, List, Dict, Any

from hugegraph_llm.models.llms.base import BaseLLM


def gremlin_examples(examples: List[Dict[str, str]]) -> str:
    example_strings = []
    for example in examples:
        example_strings.append(
            f"- query: {example['query']}\n"
            f"- gremlin: {example['gremlin']}")
    return "\n\n".join(example_strings)


def gremlin_generate_prompt(inp: str) -> str:
    return f"""Generate gremlin from the following user input.
The output format must be: "gremlin: generated gremlin".

The query is: {inp}"""


def gremlin_generate_with_schema_prompt(schema: str, inp: str) -> str:
    return f"""Given the graph schema:
{schema}
Generate gremlin from the following user input.
The output format must be: "gremlin: generated gremlin".

The query is: {inp}"""


def gremlin_generate_with_example_prompt(example: str, inp: str) -> str:
    return f"""Given the example query-gremlin pairs:
{example}

Generate gremlin from the following user input.
The output format must be: "gremlin: generated gremlin".

The query is: {inp}"""


def gremlin_generate_with_schema_and_example_prompt(schema: str, example: str, inp: str) -> str:
    return f"""Given the graph schema:
{schema}
Given the example query-gremlin pairs:
{example}

Generate gremlin from the following user input.
The output format must be: "gremlin: generated gremlin".

The query is: {inp}"""


class GremlinGenerate:
    def __init__(
            self,
            llm: BaseLLM,
            use_schema: bool = False,
            use_example: bool = False,
            schema: Optional[dict] = None
    ) -> None:
        self.llm = llm
        self.use_schema = use_schema
        self.use_example = use_example
        self.schema = schema

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        examples = context.get("match_result", [])
        if not self.use_schema and not self.use_example:
            prompt = gremlin_generate_prompt(query)
        elif not self.use_schema and self.use_example:
            prompt = gremlin_generate_with_example_prompt(gremlin_examples(examples), query)
        elif self.use_schema and not self.use_example:
            prompt = gremlin_generate_with_schema_prompt(json.dumps(self.schema), query)
        else:
            prompt = gremlin_generate_with_schema_and_example_prompt(
                json.dumps(self.schema),
                gremlin_examples(examples),
                query
            )
        response = self.llm.generate(prompt=prompt)
        context["result"] = self._extract_gremlin(response)
        return context

    def _extract_gremlin(self, response: str) -> str:
        match = re.search(r'gremlin[:：][^\n]+\n?', response)
        if match is None:
            return "Unable to generate gremlin from your query."
        return match.group()[len("gremlin:"):].strip()


if __name__ == '__main__':
    print(gremlin_examples([{"query": "hello", "gremlin": "g.V()"}]))
