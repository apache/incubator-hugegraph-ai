import re
import os
from typing import List, Dict, Any


from api.src.llm.basellm import BaseLLM
from text2kg.unstructured_data_utils import nodesTextToListOfDict, relationshipTextToListOfDict, \
    nodesschemasTextToListOfDict, relationshipschemaTextToListOfDict


def generate_system_message() -> str:
    return """
You are a data scientist working for a company that is building a graph database. Your task is to extract information from data and convert it into a graph database.
Provide a set of Nodes in the form [ENTITY_ID, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES] and a set of NodesSchemas in the form [ENTITY_TYPE, PRIMARYKEY, PROPERTIES] and a set of RelationshipsSchemas in the form [ENTITY_TYPE_1, RELATIONSHIP, ENTITY_TYPE_2, PROPERTIES]
It is important that the ENTITY_ID_1 and ENTITY_ID_2 exists as nodes with a matching ENTITY_ID. If you can't pair a relationship with a pair of nodes don't add it.
When you find a node or relationship you want to add try to create a generic TYPE for it that  describes the entity you can also think of it as a label.

Example:
Data: Alice lawyer and is 25 years old and Bob is her roommate since 2001. Bob works as a journalist. Alice owns a the webpage www.alice.com and Bob owns the webpage www.bob.com.
Nodes: ["Alice", "Person", {"age": 25, "occupation": "lawyer", "name": "Alice"}], ["Bob", "Person", {"occupation": "journalist", "name": "Bob"}], ["alice.com", "Webpage", {"name": "alice.com", "url": "www.alice.com"}], ["bob.com", "Webpage", {"name": "bob.com", "url": "www.bob.com"}]
Relationships: [{"Person": "Alice"}, "roommate", {"Person": "Bob"}, {"start": 2021}], [{"Person": "Alice"}, "owns", {"Webpage": "alice.com"}, {}], [{"Person": "Bob"}, "owns", {"Webpage": "bob.com"}, {}]
NodesSchemas: ["Person", "name",  {"age": "int", "name": "text", "occupation": "text"}],  ["Webpage", "name", {"name": "text", "url": "text"}]
RelationshipsSchemas :["Person", "roommate", "Person", {"start": "int"}], ["Person", "owns", "Webpage", {}]
"""



def generate_prompt(data) -> str:
    return f"""
Data: {data}"""



def splitString(string, max_length) -> List[str]:
    return [string[i : i + max_length] for i in range(0, len(string), max_length)]


def splitStringToFitTokenSpace(
    llm: BaseLLM, string: str, token_use_per_string: int
) -> List[str]:
    allowed_tokens = llm.max_allowed_token_length() - token_use_per_string
    chunked_data = splitString(string, 500)
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


def getNodesAndRelationshipsFromResult(result):
    regex = "Nodes:\s+(.*?)\s?\s?Relationships:\s+(.*?)\s?\s?NodesSchemas:\s+(.*?)\s?\s?\s?RelationshipsSchemas:\s?\s?(.*)"
    internalRegex = "\[(.*?)\]"
    nodes = []
    relationships = []
    nodesSchema = []
    relationshipsSchemas = []
    for row in result:
        parsing = re.match(regex, row, flags=re.S)
        if parsing == None:
            continue
        rawNodes = str(parsing.group(1))
        rawRelationships = parsing.group(2)
        rawNodesSchemas = parsing.group(3)
        rawRelationshipsSchemas = parsing.group(4)
        nodes.extend(re.findall(internalRegex, rawNodes))
        relationships.extend(re.findall(internalRegex, rawRelationships))
        nodesSchema.extend(re.findall(internalRegex, rawNodesSchemas))
        relationshipsSchemas.extend(re.findall(internalRegex, rawRelationshipsSchemas))


    result = dict()
    result["nodes"] = []
    result["relationships"] = []
    result["nodesschemas"] = []
    result["relationshipsschemas"] = []
    result["nodes"].extend(nodesTextToListOfDict(nodes))
    result["relationships"].extend(relationshipTextToListOfDict(relationships))
    result["nodesschemas"].extend(nodesschemasTextToListOfDict(nodesSchema))
    result["relationshipsschemas"].extend(relationshipschemaTextToListOfDict(relationshipsSchemas))
    print(result["nodes"])
    print(result["relationships"])
    print(result["nodesschemas"])
    print(result["relationshipsschemas"])
    return result


class TextToData():
    llm: BaseLLM

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm


    def process(self, chunk):
        messages = [
            {"role": "system", "content": generate_system_message()},
            {"role": "user", "content": generate_prompt(chunk)},
        ]
        output = self.llm.generate(messages)
        return output

    def run(self, data:str) -> dict():
        system_message = generate_system_message()
        prompt_string = generate_prompt("")
        token_usage_per_prompt = self.llm.num_tokens_from_string(
            system_message + prompt_string
        )
        chunked_data = splitStringToFitTokenSpace(
            llm=self.llm, string=data, token_use_per_string=token_usage_per_prompt
        )

        results = []
        for chunk in chunked_data:
            proceededChunk = self.process(chunk)
            results.append(proceededChunk)
            print("111111111")
            print("text2data-result: ")

        return getNodesAndRelationshipsFromResult(results)



