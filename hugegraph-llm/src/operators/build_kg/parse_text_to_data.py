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
from typing import List

from src.operators.build_kg.unstructured_data_utils import (
    nodes_text_to_list_of_dict,
    nodes_schemas_text_to_list_of_dict,
    relationships_schemas_text_to_list_of_dict,
    relationships_text_to_list_of_dict,
)
from src.operators.llm.base import BaseLLM


def generate_system_message() -> str:
    return """
You are a data scientist working for a company that is building a graph database. Your task is to extract information from data and convert it into a graph database.
Provide a set of Nodes in the form [ENTITY_ID, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES] and a set of NodesSchemas in the form [ENTITY_TYPE, PRIMARY_KEY, PROPERTIES] and a set of RelationshipsSchemas in the form [ENTITY_TYPE_1, RELATIONSHIP, ENTITY_TYPE_2, PROPERTIES]
It is important that the ENTITY_ID_1 and ENTITY_ID_2 exists as nodes with a matching ENTITY_ID. If you can't pair a relationship with a pair of nodes don't add it.
When you find a node or relationship you want to add try to create a generic TYPE for it that  describes the entity you can also think of it as a label.

Here is an example
The input you will be given:
Data: Alice lawyer and is 25 years old and Bob is her roommate since 2001. Bob works as a journalist. Alice owns a the webpage www.alice.com and Bob owns the webpage www.bob.com.
The output you need to provide:
Nodes: ["Alice", "Person", {"age": 25, "occupation": "lawyer", "name": "Alice"}], ["Bob", "Person", {"occupation": "journalist", "name": "Bob"}], ["alice.com", "Webpage", {"name": "alice.com", "url": "www.alice.com"}], ["bob.com", "Webpage", {"name": "bob.com", "url": "www.bob.com"}]
Relationships: [{"Person": "Alice"}, "roommate", {"Person": "Bob"}, {"start": 2021}], [{"Person": "Alice"}, "owns", {"Webpage": "alice.com"}, {}], [{"Person": "Bob"}, "owns", {"Webpage": "bob.com"}, {}]
NodesSchemas: ["Person", "name",  {"age": "int", "name": "text", "occupation": "text"}],  ["Webpage", "name", {"name": "text", "url": "text"}]
RelationshipsSchemas :["Person", "roommate", "Person", {"start": "int"}], ["Person", "owns", "Webpage", {}]
"""


def generate_system_message_with_schemas() -> str:
    return """
You are a data scientist working for a company that is building a graph database. Your task is to extract information from data and convert it into a graph database.
Provide a set of Nodes in the form [ENTITY_ID, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES] and a set of NodesSchemas in the form [ENTITY_TYPE, PRIMARY_KEY, PROPERTIES] and a set of RelationshipsSchemas in the form [ENTITY_TYPE_1, RELATIONSHIP, ENTITY_TYPE_2, PROPERTIES]
It is important that the ENTITY_ID_1 and ENTITY_ID_2 exists as nodes with a matching ENTITY_ID. If you can't pair a relationship with a pair of nodes don't add it.
When you find a node or relationship you want to add try to create a generic TYPE for it that  describes the entity you can also think of it as a label.

Here is an example
The input you will be given:
Data: Alice lawyer and is 25 years old and Bob is her roommate since 2001. Bob works as a journalist. Alice owns a the webpage www.alice.com and Bob owns the webpage www.bob.com.
NodesSchemas: ["Person", "name",  {"age": "int", "name": "text", "occupation": "text"}],  ["Webpage", "name", {"name": "text", "url": "text"}]
RelationshipsSchemas :["Person", "roommate", "Person", {"start": "int"}], ["Person", "owns", "Webpage", {}]
The output you need to provide:
Nodes: ["Alice", "Person", {"age": 25, "occupation": "lawyer", "name": "Alice"}], ["Bob", "Person", {"occupation": "journalist", "name": "Bob"}], ["alice.com", "Webpage", {"name": "alice.com", "url": "www.alice.com"}], ["bob.com", "Webpage", {"name": "bob.com", "url": "www.bob.com"}]
Relationships: [{"Person": "Alice"}, "roommate", {"Person": "Bob"}, {"start": 2021}], [{"Person": "Alice"}, "owns", {"Webpage": "alice.com"}, {}], [{"Person": "Bob"}, "owns", {"Webpage": "bob.com"}, {}]
NodesSchemas: ["Person", "name",  {"age": "int", "name": "text", "occupation": "text"}],  ["Webpage", "name", {"name": "text", "url": "text"}]
RelationshipsSchemas :["Person", "roommate", "Person", {"start": "int"}], ["Person", "owns", "Webpage", {}]
"""


def generate_prompt(data) -> str:
    return f"""
Data: {data}"""


def generate_prompt_with_schemas(data, nodes_schemas, relationships_schemas) -> str:
    return f"""
Data: {data}
NodesSchemas: {nodes_schemas}
RelationshipsSchemas: {relationships_schemas}"""


def split_string(string, max_length) -> List[str]:
    return [string[i : i + max_length] for i in range(0, len(string), max_length)]


def split_string_to_fit_token_space(
    llm: BaseLLM, string: str, token_use_per_string: int
) -> List[str]:
    allowed_tokens = llm.max_allowed_token_length() - token_use_per_string
    chunked_data = split_string(string, 500)
    combined_chunks = []
    current_chunk = ""
    for chunk in chunked_data:
        if (
            llm.num_tokens_from_string(current_chunk)
            + llm.num_tokens_from_string(chunk)
            < allowed_tokens
        ):
            current_chunk += chunk
        else:
            combined_chunks.append(current_chunk)
            current_chunk = chunk
    combined_chunks.append(current_chunk)

    return combined_chunks


def get_nodes_and_relationships_from_result(result):
    regex = (
        r"Nodes:\s+(.*?)\s?\s?Relationships:\s+(.*?)\s?\s?NodesSchemas:\s+(.*?)\s?\s?\s?"
        r"RelationshipsSchemas:\s?\s?(.*)"
    )
    internal_regex = r"\[(.*?)\]"
    nodes = []
    relationships = []
    nodes_schemas = []
    relationships_schemas = []
    for row in result:
        parsing = re.match(regex, row, flags=re.S)
        if parsing is None:
            continue
        raw_nodes = str(parsing.group(1))
        raw_relationships = parsing.group(2)
        raw_nodes_schemas = parsing.group(3)
        raw_relationships_schemas = parsing.group(4)
        nodes.extend(re.findall(internal_regex, raw_nodes))
        relationships.extend(re.findall(internal_regex, raw_relationships))
        nodes_schemas.extend(re.findall(internal_regex, raw_nodes_schemas))
        relationships_schemas.extend(
            re.findall(internal_regex, raw_relationships_schemas)
        )
    result = dict()
    result["nodes"] = []
    result["relationships"] = []
    result["nodes_schemas"] = []
    result["relationships_schemas"] = []
    result["nodes"].extend(nodes_text_to_list_of_dict(nodes))
    result["relationships"].extend(relationships_text_to_list_of_dict(relationships))
    result["nodes_schemas"].extend(nodes_schemas_text_to_list_of_dict(nodes_schemas))
    result["relationships_schemas"].extend(
        relationships_schemas_text_to_list_of_dict(relationships_schemas)
    )
    return result


class ParseTextToData:
    llm: BaseLLM

    def __init__(self, llm: BaseLLM, text: str) -> None:
        self.llm = llm
        self.text = text

    def process(self, chunk):
        messages = [
            {"role": "system", "content": generate_system_message()},
            {"role": "user", "content": generate_prompt(chunk)},
        ]

        output = self.llm.generate(messages)
        return output

    def run(self, data: dict) -> dict[str, list[any]]:
        system_message = generate_system_message()
        prompt_string = generate_prompt("")
        token_usage_per_prompt = self.llm.num_tokens_from_string(
            system_message + prompt_string
        )
        chunked_data = split_string_to_fit_token_space(
            llm=self.llm, string=self.text, token_use_per_string=token_usage_per_prompt
        )

        results = []
        for chunk in chunked_data:
            proceeded_chunk = self.process(chunk)
            results.append(proceeded_chunk)
            results = get_nodes_and_relationships_from_result(results)

        return results


class ParseTextToDataWithSchemas:
    llm: BaseLLM

    def __init__(
        self, llm: BaseLLM, text: str, nodes_schema, relationships_schemas
    ) -> None:
        self.llm = llm
        self.text = text
        self.data = {}
        self.nodes_schemas = nodes_schema
        self.relationships_schemas = relationships_schemas

    def process_with_schemas(self, chunk):
        messages = [
            {"role": "system", "content": generate_system_message_with_schemas()},
            {
                "role": "user",
                "content": generate_prompt_with_schemas(
                    chunk, self.nodes_schemas, self.relationships_schemas
                ),
            },
        ]

        output = self.llm.generate(messages)
        return output

    def run(self) -> dict[str, list[any]]:
        system_message = generate_system_message_with_schemas()
        prompt_string = generate_prompt_with_schemas("", "", "")
        token_usage_per_prompt = self.llm.num_tokens_from_string(
            system_message + prompt_string
        )
        chunked_data = split_string_to_fit_token_space(
            llm=self.llm, string=self.text, token_use_per_string=token_usage_per_prompt
        )

        results = []
        for chunk in chunked_data:
            proceeded_chunk = self.process_with_schemas(chunk)
            results.append(proceeded_chunk)
            results = get_nodes_and_relationships_from_result(results)
        return results
