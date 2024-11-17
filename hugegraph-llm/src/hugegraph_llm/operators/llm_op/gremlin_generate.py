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
from typing import Optional, List, Dict, Any, Union

from hugegraph_llm.models.llms.base import BaseLLM


def gremlin_examples(examples: List[Dict[str, str]]) -> str:
    example_strings = []
    for example in examples:
        example_strings.append(
            f"- query: {example['query']}\n"
            f"- gremlin:\n```gremlin\n{example['gremlin']}\n```")
    return "\n\n".join(example_strings)


def gremlin_generate_prompt(query: str) -> str:
    return f"""\
Generate gremlin from the following user query.
The output format must be like:
```gremlin
g.V().limit(10)
```

Generate gremlin from the following user query.
{query}
The generated gremlin is:"""


def gremlin_generate_with_schema_prompt(schema: str, query: str) -> str:
    return f"""\
Given the graph schema:
{schema}
Generate gremlin from the following user query.
The output format must be like:
```gremlin
g.V().limit(10)
```

Generate gremlin from the following user query.
{query}
The generated gremlin is:"""


def gremlin_generate_with_example_prompt(example: str, query: str) -> str:
    return f"""Given the example query-gremlin pairs:
{example}

Generate gremlin from the following user query.
The output format must be like:
```gremlin
g.V().limit(10)
```

Generate gremlin from the following user query.
{query}
The generated gremlin is:"""


def gremlin_generate_with_schema_and_example_prompt(schema: str, example: str, query: str) -> str:
    return f"""\
Given the example query-gremlin pairs:
{example}

Given the graph schema:
```json
{schema}
```
Generate gremlin from the following user query.
{query}
The generated gremlin is:"""


class GremlinGenerate:
    def __init__(
            self,
            llm: BaseLLM,
            use_schema: bool = False,
            use_example: bool = False,
            schema: Optional[Union[dict, str]] = None
    ) -> None:
        self.llm = llm
        self.use_schema = use_schema
        self.use_example = use_example
        if isinstance(schema, dict):
            schema = json.dumps(schema, encode='utf8')
        self.schema = schema

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        assert query, "query is required"
        examples = context.get("match_result", [])
        if not self.use_schema and not self.use_example:
            prompt = gremlin_generate_prompt(query)
        elif not self.use_schema and self.use_example:
            prompt = gremlin_generate_with_example_prompt(gremlin_examples(examples), query)
        elif self.use_schema and not self.use_example:
            prompt = gremlin_generate_with_schema_prompt(self.schema, query)
        else:
            prompt = gremlin_generate_with_schema_and_example_prompt(
                self.schema,
                gremlin_examples(examples),
                query
            )
        response = self.llm.generate(prompt=prompt)
        context["result"] = self._extract_gremlin(response)

        context["call_count"] = context.get("call_count", 0) + 1
        return context

    def _extract_gremlin(self, response: str) -> str:
        match = re.search("```gremlin.*```", response, re.DOTALL)
        if match is None:
            return "Unable to generate gremlin from your query."
        return match.group()[len("```gremlin"):-len("```")].strip()


if __name__ == '__main__':
    print(gremlin_examples([{"query": "hello", "gremlin": "g.V()"}]))
