import json
import re

regex = "Nodes:\s+(.*?)\s?\s?Relationships:\s?\s?NodesSchemas:\s+(.*?)\s?\s?RelationshipsSchemas:\s?\s?(.*)"
internalRegex = "\[(.*?)\]"
jsonRegex = "\{.*\}"
jsonRegex_relationships = "\{.*?\}"


def nodesTextToListOfDict(nodes):
    result = []
    for node in nodes:
        nodeList = node.split(",")
        if len(nodeList) < 2:
            continue

        name = nodeList[0].strip().replace('"', "")
        label = nodeList[1].strip().replace('"', "")
        properties = re.search(jsonRegex, node)
        if properties == None:
            properties = "{}"
        else:
            properties = properties.group(0)
        properties = properties.replace("True", "true")
        try:
            properties = json.loads(properties)
        except:
            properties = {}
        result.append({"name": name, "label": label, "properties": properties})
    return result


def relationshipTextToListOfDict(relationships):
    result = []
    for relation in relationships:
        relationList = relation.split(",")
        if len(relationList) < 3:
            continue
        start = {}
        end = {}
        properties = {}
        type = relationList[1].strip().replace('"', "")
        matches = re.findall(jsonRegex_relationships, relation)
        i = 1
        for match in matches:
            if i == 1:
                start = json.loads(match)
                i = 2
                continue
            if i == 2:
                end = json.loads(match)
                i = 3
                continue
            if i == 3:
                properties = json.loads(match)
        result.append(
            {"start": start, "end": end, "type": type, "properties": properties}
        )
    return result

def nodesschemasTextToListOfDict(nodes_schemas):
    result = []
    for node_scheam in nodes_schemas:
        node_scheamList = node_scheam.split(",")
        if len(node_scheam) < 1:
            continue

        label = node_scheamList[0].strip().replace('"', "")
        primaryKey = node_scheamList[1].strip().replace('"', "")
        properties = re.search(jsonRegex, node_scheam)
        if properties == None:
            properties = "{}"
        else:
            properties = properties.group(0)
        properties = properties.replace("True", "true")
        try:
            properties = json.loads(properties)
        except:
            properties = {}
        result.append({"label": label, "primaryKey":primaryKey, "properties": properties})
    return result


def relationshipschemaTextToListOfDict(schemas):
    result = []
    for schema in schemas:
        schemaList = schema.split(",")
        if len(schemaList) < 3:
            continue
        start = schemaList[0].strip().replace('"', "")
        end = schemaList[2].strip().replace('"', "")
        type = schemaList[1].strip().replace('"', "")

        properties = re.search(jsonRegex, schema)
        if properties == None:
            properties = "{}"
        else:
            properties = properties.group(0)
        properties = properties.replace("True", "true")
        try:
            properties = json.loads(properties)
        except:
            properties = {}
        result.append(
            {"start": start, "end": end, "type": type, "properties": properties}
        )
    return result

