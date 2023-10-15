import os
from itertools import groupby

from hugegraph.connection import PyHugeGraph



def generate_new_relationships(nodes_schemas_data, relationships_data):
    labelId = dict()
    i = 1
    old_label = []
    for item in nodes_schemas_data:
        label = item["label"]
        if label in old_label:
            continue
        else:
            labelId[label] = i
            i += 1
            old_label.append(label)
    new_relationships_data = []

    for relationship in relationships_data:
        start = relationship['start']
        end = relationship['end']
        type = relationship['type']
        properties = relationship['properties']
        new_start = []
        new_end = []
        for key, value in labelId.items():
            for key1, value1 in start.items():
                if key1 == key:
                    new_start = f'{value}' + ':' + f'{value1}'
            for key1, value1 in end.items():
                if key1 == key:
                    new_end = f'{value}' + ':' + f'{value1}'
        relationships_data = dict()
        relationships_data["start"] = new_start
        relationships_data["end"] = new_end
        relationships_data["type"] = type
        relationships_data["properties"] = properties
        new_relationships_data.append(relationships_data)
    return new_relationships_data


def generate_schema_properties(data):
    schema_properties_statements = []
    if len(data) == 3:
        for item in data:
            properties = item['properties']
            for key, value in properties.items():
                if value == 'int':
                    schema_properties_statements.append(f"schema.propertyKey('{key}').asInt().ifNotExist().create()")
                elif value == 'text':
                    schema_properties_statements.append(f"schema.propertyKey('{key}').asText().ifNotExist().create()")
    else:
        for item in data:
            properties = item['properties']
            for key, value in properties.items():
                if value == 'int':
                    schema_properties_statements.append(f"schema.propertyKey('{key}').asInt().ifNotExist().create()")
                elif value == 'text':
                    schema_properties_statements.append(f"schema.propertyKey('{key}').asText().ifNotExist().create()")
    return schema_properties_statements


def generate_schema_nodes(data):
    schema_nodes_statements = []
    for item in data:
        label = item['label']
        primaryKey = item['primaryKey']
        properties = item['properties']

        schema_statement = f"schema.vertexLabel('{label}').properties("
        schema_statement += ', '.join(f"'{prop}'" for prop in properties.keys())
        schema_statement += f").nullableKeys("
        schema_statement += ', '.join(f"'{prop}'" for prop in properties.keys() if prop != primaryKey)
        schema_statement += f").usePrimaryKeyId().primaryKeys('{primaryKey}').ifNotExist().create()"
        schema_nodes_statements.append(schema_statement)
    return schema_nodes_statements


def generate_schema_relationships(data):
    schema_relstionships_statements = []
    for item in data:
        start = item['start']
        end = item['end']
        type = item['type']
        properties = item['properties']
        schema_statement = f"schema.edgeLabel('{type}').sourceLabel('{start}').targetLabel('{end}').properties("
        schema_statement += ', '.join(f"'{prop}'" for prop in properties.keys())
        schema_statement += f").nullableKeys("
        schema_statement += ', '.join(f"'{prop}'" for prop in properties.keys())
        schema_statement += f").ifNotExist().create()"
        schema_relstionships_statements.append(schema_statement)
    return schema_relstionships_statements


def generate_nodes(data):
    nodes = []
    for item in data:
        label = item['label']
        properties = item['properties']
        nodes.append(f"g.addVertex('{label}', {properties})")
    return nodes


def generate_relationships(data):
    relationships = []
    for item in data:
        start = item['start']
        end = item['end']
        type = item['type']
        properties = item['properties']
        relationships.append(f"g.addEdge('{type}', '{start}', '{end}', {properties})")
    return relationships


class DataToKg():
    def __init__(self):
        self.client = PyHugeGraph("127.0.0.1", "8080", user="admin", pwd="admin", graph="hugegraph")
        self.schema = self.client.schema()

    def run(self, data: dict):
        os.environ.pop("http_proxy")
        os.environ.pop("https_proxy")
        nodes = data["nodes"]
        relationships = data["relationships"]
        nodes_schemas = data["nodesschemas"]
        relationships_schemas = data["relationshipsschemas"]
        schema = self.schema
        # properties schema
        schema_nodes_properties = generate_schema_properties(nodes_schemas)
        schema_relationships_properties = generate_schema_properties(relationships_schemas)
        for schema_nodes_property in schema_nodes_properties:
            print(schema_nodes_property)
            exec(schema_nodes_property)

        for schema_relationships_property in schema_relationships_properties:
            print(schema_relationships_property)
            exec(schema_relationships_property)

        # nodes schema
        schema_nodes = generate_schema_nodes(nodes_schemas)
        for schema_node in schema_nodes:
            print(schema)
            exec(schema_node)

        # relationships schema
        schema_relationships = generate_schema_relationships(relationships_schemas)
        for schema_relationship in schema_relationships:
            print(schema_relationship)
            exec(schema_relationship)

        g = self.client.graph()
        # nodes
        nodes = generate_nodes(nodes)
        for node in nodes:
            print(node)
            exec(node)

        # relationships
        new_relationships = generate_new_relationships(nodes_schemas, relationships)
        relationships_schemas = generate_relationships(new_relationships)
        for relationship in relationships_schemas:
            print(relationship)
            exec(relationship)


#