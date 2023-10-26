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


import json
import re
from itertools import groupby
from typing import Dict, List, Any

from hugegraph_llm.operators.llm_op.unstructured_data_utils import (
    nodes_text_to_list_of_dict,
    relationships_text_to_list_of_dict,
    relationships_schemas_text_to_list_of_dict,
    nodes_schemas_text_to_list_of_dict,
)
from hugegraph_llm.llms.base import BaseLLM


def disambiguate_nodes() -> str:
    return """
Your task is to identify if there are duplicated nodes and if so merge them into one nod. Only merge the nodes that refer to the same entity.
You will be given different datasets of nodes and some of these nodes may be duplicated or refer to the same entity. 
The datasets contains nodes in the form [ENTITY_ID, TYPE, PROPERTIES]. When you have completed your task please give me the 
resulting nodes in the same format. Only return the nodes and relationships no other text. If there is no duplicated nodes return the original nodes.

Here is an example
The input you will be given:
["Alice", "Person", {"age" : 25, "occupation": "lawyer", "name":"Alice"}], ["Bob", "Person", {"occupation": "journalist", "name": "Bob"}], ["alice.com", "Webpage", {"url": "www.alice.com"}], ["bob.com", "Webpage", {"url": "www.bob.com"}], ["Bob", "Person", {"occupation": "journalist", "name": "Bob"}]
The output you need to provide:
["Alice", "Person", {"age" : 25, "occupation": "lawyer", "name":"Alice"}], ["Bob", "Person", {"occupation": "journalist", "name": "Bob"}], ["alice.com", "Webpage", {"url": "www.alice.com"}], ["bob.com", "Webpage", {"url": "www.bob.com"}]
"""


def disambiguate_relationships() -> str:
    return """
Your task is to identify if a set of relationships make sense.
If they do not make sense please remove them from the dataset.
Some relationships may be duplicated or refer to the same entity. 
Please merge relationships that refer to the same entity.
The datasets contains relationships in the form [{"ENTITY_TYPE_1": "ENTITY_ID_1"}, RELATIONSHIP, {"ENTITY_TYPE_2": "ENTITY_ID_2"}, PROPERTIES].
You will also be given a set of ENTITY_IDs that are valid.
Some relationships may use ENTITY_IDs that are not in the valid set but refer to a entity in the valid set.
If a relationships refer to a ENTITY_ID in the valid set please change the ID so it matches the valid ID.
When you have completed your task please give me the valid relationships in the same format. Only return the relationships no other text.

Here is an example
The input you will be given:
[{"Person": "Alice"}, "roommate", {"Person": "bob"}, {"start": 2021}], [{"Person": "Alice"}, "owns", {"Webpage": "alice.com"}, {}], [{"Person": "Bob"}, "owns", {"Webpage": "bob.com"}, {}], [{"Person": "Alice"}, "owns", {"Webpage": "alice.com"}, {}]
The output you need to provide:
[{"Person": "Alice"}, "roommate", {"Person": "bob"}, {"start": 2021}], [{"Person": "Alice"}, "owns", {"Webpage": "alice.com"}, {}], [{"Person": "Bob"}, "owns", {"Webpage": "bob.com"}, {}]
"""


def disambiguate_nodes_schemas() -> str:
    return """
Your task is to identify if there are duplicated nodes schemas and if so merge them into one nod. Only merge the nodes schemas that refer to the same entty_types.
You will be given different node schemas, some of which may duplicate or reference the same entty_types. Note: For node schemas with the same entty_types, you need to merge them while merging all properties of the entty_types. 
The datasets contains nodes schemas in the form [ENTITY_TYPE, PRIMARY KEY, PROPERTIES]. When you have completed your task please give me the 
resulting nodes schemas in the same format. Only return the nodes schemas no other text. If there is no duplicated nodes return the original nodes schemas.

Here is an example
The input you will be given:
["Person", "name",  {"age": "int", "name": "text", "occupation": "text"}],  ["Webpage", "url", {url: "text"}],  ["Webpage", "url", {url: "text"}]
The output you need to provide:
["Person", "name",  {"age": "int", "name": "text", "occupation": "text"}],  ["Webpage", "url", {url: "text"}]
"""


def disambiguate_relationships_schemas() -> str:
    return """
Your task is to identify if a set of relationships schemas make sense.
If they do not make sense please remove them from the dataset.
Some relationships may be duplicated or refer to the same label. 
Please merge relationships that refer to the same label.
The datasets contains relationships in the form [LABEL_ID_1, RELATIONSHIP, LABEL_ID_2, PROPERTIES].
You will also be given a set of LABELS_IDs that are valid.
Some relationships may use LABELS_IDs that are not in the valid set but refer to a LABEL in the valid set.
If a relationships refer to a LABELS_IDs in the valid set please change the ID so it matches the valid ID.
When you have completed your task please give me the valid relationships in the same format. Only return the relationships no other text.

Here is an example
["Person", "roommate", "Person", {"start": 2021}], ["Person", "owns", "Webpage", {}], ["Person", "roommate", "Person", {"start": 2021}]
The output you need to provide:
["Person", "roommate", "Person", {"start": 2021}], ["Person", "owns", "Webpage", {}]
"""


def generate_prompt(data) -> str:
    return f""" Here is the data:
{data}
"""


INTERNAL_REGEX = r"\[(.*?)\]"


class DisambiguateData:
    def __init__(self, llm: BaseLLM, is_user_schema: bool) -> None:
        self.llm = llm
        self.is_user_schema = is_user_schema

    def run(self, data: Dict) -> Dict[str, List[Any]]:
        nodes = sorted(data["nodes"], key=lambda x: x.get("label", ""))
        relationships = data["relationships"]
        nodes_schemas = data["nodes_schemas"]
        relationships_schemas = data["relationships_schemas"]
        new_nodes = []
        new_relationships = []
        new_nodes_schemas = []
        new_relationships_schemas = []

        node_groups = groupby(nodes, lambda x: x["label"])
        for group in node_groups:
            dis_string = ""
            nodes_in_group = list(group[1])
            if len(nodes_in_group) == 1:
                new_nodes.extend(nodes_in_group)
                continue

            for node in nodes_in_group:
                dis_string += (
                    '["'
                    + node["name"]
                    + '", "'
                    + node["label"]
                    + '", '
                    + json.dumps(node["properties"])
                    + "]\n"
                )

            messages = [
                {"role": "system", "content": disambiguate_nodes()},
                {"role": "user", "content": generate_prompt(dis_string)},
            ]
            raw_nodes = self.llm.generate(messages)
            n = re.findall(INTERNAL_REGEX, raw_nodes)
            new_nodes.extend(nodes_text_to_list_of_dict(n))

        relationship_data = ""
        for relation in relationships:
            relationship_data += (
                '["'
                + json.dumps(relation["start"])
                + '", "'
                + relation["type"]
                + '", "'
                + json.dumps(relation["end"])
                + '", '
                + json.dumps(relation["properties"])
                + "]\n"
            )

        node_labels = [node["name"] for node in new_nodes]
        relationship_data += "Valid Nodes:\n" + "\n".join(node_labels)

        messages = [
            {
                "role": "system",
                "content": disambiguate_relationships(),
            },
            {"role": "user", "content": generate_prompt(relationship_data)},
        ]
        raw_relationships = self.llm.generate(messages)
        rels = re.findall(INTERNAL_REGEX, raw_relationships)
        new_relationships.extend(relationships_text_to_list_of_dict(rels))

        if not self.is_user_schema:
            nodes_schemas_data = ""
            for node_schema in nodes_schemas:
                nodes_schemas_data += (
                    '["'
                    + node_schema["label"]
                    + '", '
                    + node_schema["primary_key"]
                    + '", '
                    + json.dumps(node_schema["properties"])
                    + "]\n"
                )

            messages = [
                {"role": "system", "content": disambiguate_nodes_schemas()},
                {"role": "user", "content": generate_prompt(nodes_schemas_data)},
            ]
            raw_nodes_schemas = self.llm.generate(messages)
            n = re.findall(INTERNAL_REGEX, raw_nodes_schemas)
            new_nodes_schemas.extend(nodes_schemas_text_to_list_of_dict(n))

            relationships_schemas_data = ""
            for relationships_schema in relationships_schemas:
                relationships_schemas_data += (
                    '["'
                    + relationships_schema["start"]
                    + '", "'
                    + relationships_schema["type"]
                    + '", "'
                    + relationships_schema["end"]
                    + '", '
                    + json.dumps(relationships_schema["properties"])
                    + "]\n"
                )

            node_schemas_labels = [nodes_schemas["label"] for nodes_schemas in new_nodes_schemas]
            relationships_schemas_data += "Valid Labels:\n" + "\n".join(node_schemas_labels)

            messages = [
                {
                    "role": "system",
                    "content": disambiguate_relationships_schemas(),
                },
                {
                    "role": "user",
                    "content": generate_prompt(relationships_schemas_data),
                },
            ]
            raw_relationships_schemas = self.llm.generate(messages)
            schemas_rels = re.findall(INTERNAL_REGEX, raw_relationships_schemas)
            new_relationships_schemas.extend(
                relationships_schemas_text_to_list_of_dict(schemas_rels)
            )
        else:
            new_nodes_schemas = nodes_schemas
            new_relationships_schemas = relationships_schemas

        return {
            "nodes": new_nodes,
            "relationships": new_relationships,
            "nodes_schemas": new_nodes_schemas,
            "relationships_schemas": new_relationships_schemas,
        }
