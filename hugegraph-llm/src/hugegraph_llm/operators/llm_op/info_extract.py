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
from typing import List, Any, Dict

from hugegraph_llm.llms.base import BaseLLM


def generate_extract_triple_prompt(text, schema=None) -> str:
    if schema:
        return f"""
        Given the graph schema: {schema}
        
        Based on the above schema, extract triples from the following text.
        The output format must be: (X,Y,Z) - LABEL
        In this format, Y must be a value from "properties" or "edge_label", 
        and LABEL must be X's vertex_label or Y's edge_label.
        
        The extracted text is: {text}
        """
    return f"""Extract subject-verb-object (SPO) triples from text strictly according to the
        following format, each structure has only three elements: ("vertex_1", "edge", "vertex_2").
        for example:
        Alice lawyer and is 25 years old and Bob is her roommate since 2001. Bob works as a journalist. 
        Alice owns a the webpage www.alice.com and Bob owns the webpage www.bob.com
        output:[("Alice", "Age", "25"),("Alice", "Profession", "lawyer"),("Bob", "Job", "journalist"),
        ("Alice", "Roommate of", "Bob"),("Alice", "Owns", "http://www.alice.com"),
        ("Bob", "Owns", "http://www.bob.com")]

        The extracted text is: {text}"""


def fit_token_space_by_split_text(llm: BaseLLM, text: str, prompt_token: int) -> List[str]:
    max_length = 500
    allowed_tokens = llm.max_allowed_token_length() - prompt_token
    chunked_data = [text[i : i + max_length] for i in range(0, len(text), max_length)]
    combined_chunks = []
    current_chunk = ""
    for chunk in chunked_data:
        if (
            int(llm.num_tokens_from_string(current_chunk)) + int(llm.num_tokens_from_string(chunk))
            < allowed_tokens
        ):
            current_chunk += chunk
        else:
            combined_chunks.append(current_chunk)
            current_chunk = chunk
    combined_chunks.append(current_chunk)

    return combined_chunks


def extract_by_regex(text, triples):
    text = text.replace("\\n", " ").replace("\\", " ").replace("\n", " ")
    pattern = r"\((.*?), (.*?), (.*?)\)"
    triples["triples"] += re.findall(pattern, text)


def extract_by_regex_with_schema(schema, text, graph):
    text = text.replace("\\n", " ").replace("\\", " ").replace("\n", " ")
    pattern = r"\((.*?), (.*?), (.*?)\) - ([^ ]*)"
    matches = re.findall(pattern, text)

    vertices_dict = {}
    for match in matches:
        s, p, o, label = [item.strip() for item in match]
        if None in [label, s, p, o]:
            continue
        for vertex in schema["vertices"]:
            if vertex["vertex_label"] == label and p in vertex["properties"]:
                if (s, label) not in vertices_dict:
                    vertices_dict[(s, label)] = {"name": s, "label": label, "properties": {p: o}}
                else:
                    vertices_dict[(s, label)]["properties"].update({p: o})
                break
        for edge in schema["edges"]:
            if edge["edge_label"] == label:
                graph["edges"].append({"start": s, "end": o, "type": label, "properties": {}})
                break
    graph["vertices"] = list(vertices_dict.values())


class InfoExtract:
    def __init__(
        self,
        llm: BaseLLM,
        text: str,
    ) -> None:
        self.llm = llm
        self.text = text

    def run(self, schema=None) -> Dict[str, List[Any]]:
        prompt_token = self.llm.num_tokens_from_string(generate_extract_triple_prompt("", schema))

        chunked_text = fit_token_space_by_split_text(
            llm=self.llm, text=self.text, prompt_token=int(prompt_token)
        )

        result = {"vertices": [], "edges": [], "schema": schema} if schema else {"triples": []}
        for chunk in chunked_text:
            proceeded_chunk = self.extract_by_llm(schema, chunk)
            print(f"[LLM] input: {chunk} \n output:{proceeded_chunk}")
            if schema:
                extract_by_regex_with_schema(schema, proceeded_chunk, result)
            else:
                extract_by_regex(proceeded_chunk, result)
        return result

    def extract_by_llm(self, schema, chunk):
        prompt = generate_extract_triple_prompt(chunk, schema)
        return self.llm.generate(prompt=prompt)
