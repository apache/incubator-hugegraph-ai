import json
import re
import time
from itertools import groupby

from text2kg.unstructured_data_utils import nodesTextToListOfDict, nodesschemasTextToListOfDict, \
    relationshipTextToListOfDict, relationshipschemaTextToListOfDict


def generate_system_message_for_nodes() -> str:
    return """Your task is to identify if there are duplicated nodes and if so merge them into one nod. Only merge the nodes that refer to the same entity.
You will be given different datasets of nodes and some of these nodes may be duplicated or refer to the same entity. 
The datasets contains nodes in the form [ENTITY_ID, TYPE, PROPERTIES]. When you have completed your task please give me the 
resulting nodes in the same format. Only return the nodes and relationships no other text. If there is no duplicated nodes return the original nodes.

Here is an example of the input you will be given:
["Alice", "Person", {"age" : 25, "occupation": "lawyer", "name":"Alice"}], ["Bob", "Person", {"occupation": "journalist", "name": "Bob"}], ["alice.com", "Webpage", {"url": "www.alice.com"}], ["bob.com", "Webpage", {"url": "www.bob.com"}]
"""


def generate_system_message_for_relationships() -> str:
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

Here is an example of the input you will be given:
[{"Person": "Alice"}, "roommate", {"Person": "bob"}, {"start": 2021}], [{"Person": "Alice"}, "owns", {"Webpage": "alice.com"}, {}], [{"Person": "Bob"}, "owns", {"Webpage": "bob.com"}, {}]
"""


def generate_system_message_for_nodesSchemas() -> str:
    return """Your task is to identify if there are duplicated nodes schemas and if so merge them into one nod. Only merge the nodes schemas that refer to the same entty_types.
You will be given different node schemas, some of which may duplicate or reference the same entty_types. Note: For node schemas with the same entty_types, you need to merge them while merging all properties of the entty_types. 
The datasets contains nodes schemas in the form [ENTITY_TYPE, PRIMARYKEY, PROPERTIES]. When you have completed your task please give me the 
resulting nodes schemas in the same format. Only return the nodes schemas no other text. If there is no duplicated nodes return the original nodes schemas.

Here is an example of the input you will be given:
["Person", "name",  {"age": "int", "name": "text", "occupation": "text"}],  ["Webpage", "url", {url: "text"}]
The output:
["Person", "name",  {"age": "int", "name": "text", "occupation": "text"}],  ["Webpage", "url", {url: "text"}]
"""


def generate_system_message_for_relationshipsSchemas() -> str:
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

Here is an example of the input you will be given:
["Person", "roommate", "Person", {"start": 2021}], ["Person", "owns", "Webpage", {}]
"""


def generate_prompt(data) -> str:
    return f""" Here is the data:
{data}
"""


internalRegex = "\[(.*?)\]"


class DataDisambiguation():
    def __init__(self, llm) -> None:
        self.llm = llm

    def run(self, data: dict) -> dict[str, list[any]]:
        nodes = sorted(data["nodes"], key=lambda x: x.get("label", ""))
        relationships = data["relationships"]
        nodes_schemas = data["nodesschemas"]
        relationships_schemas = data["relationshipsschemas"]
        new_nodes = []
        new_relationships = []
        new_nodes_schemas = []
        new_relationships_schemas = []

        node_groups = groupby(nodes, lambda x: x["label"])
        for group in node_groups:
            disString = ""
            nodes_in_group = list(group[1])
            if len(nodes_in_group) == 1:
                new_nodes.extend(nodes_in_group)
                continue

            for node in nodes_in_group:
                disString += (
                    '["'
                    + node["name"]
                    + '", "'
                    + node["label"]
                    + '", '
                    + json.dumps(node["properties"])
                    + "]\n"
                )

            messages = [
                {"role": "system", "content": generate_system_message_for_nodes()},
                {"role": "user", "content": generate_prompt(disString)},
            ]
            rawNodes = self.llm.generate(messages)

            n = re.findall(internalRegex, rawNodes)

            new_nodes.extend(nodesTextToListOfDict(n))

        time.sleep(20)

        nodes_schemas_data = ""
        for node_schema in nodes_schemas:
            nodes_schemas_data += (
                    '["'
                    + node_schema["label"]
                    + '", '
                    + node_schema["primaryKey"]
                    + '", '
                    + json.dumps(node_schema["properties"])
                    + "]\n"
            )

        messages = [
            {"role": "system", "content": generate_system_message_for_nodesSchemas()},
            {"role": "user", "content": generate_prompt(nodes_schemas_data)},
        ]
        rawNodesSchemas = self.llm.generate(messages)

        n = re.findall(internalRegex, rawNodesSchemas)

        new_nodes_schemas.extend(nodesschemasTextToListOfDict(n))

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
                "content": generate_system_message_for_relationships(),
            },
            {"role": "user", "content": generate_prompt(relationship_data)},
        ]
        rawRelationships = self.llm.generate(messages)
        rels = re.findall(internalRegex, rawRelationships)
        new_relationships.extend(relationshipTextToListOfDict(rels))


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
                "content": generate_system_message_for_relationshipsSchemas(),
            },
            {"role": "user", "content": generate_prompt(relationships_schemas_data)},
        ]
        rawRelationshipsSchema = self.llm.generate(messages)
        schemaRels = re.findall(internalRegex, rawRelationshipsSchema)
        new_relationships_schemas.extend(relationshipschemaTextToListOfDict(schemaRels))

        print(2)
        print("data2data-result: ")
        print(new_nodes)
        print(new_relationships)
        print(new_nodes_schemas)
        print(new_relationships_schemas)
        return {"nodes": new_nodes, "relationships": new_relationships, "nodesschemas": new_nodes_schemas, "relationshipsschemas": new_relationships_schemas}
