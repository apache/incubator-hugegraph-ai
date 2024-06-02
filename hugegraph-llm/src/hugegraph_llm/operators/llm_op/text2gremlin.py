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

from hugegraph_llm.llms.base import BaseLLM


def generate_text2gremlin_prompt(query):
    return f"""
        Your task is to convert the given text to a Gremlin query.

        For example:
        Text: "Get all the people who live in New York City."
        Gremlin Query:
        g.V().hasLabel('person').has('location', 'New York City')

        Please output only the Gremlin query without any additional text.
        Text: {query}
        Gremlin Query:
        """


class Text2Gremlin:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def run(self, text: str) -> str:
        prompt = generate_text2gremlin_prompt(text)
        llm_output = self.llm.generate(prompt=prompt)
        return llm_output
