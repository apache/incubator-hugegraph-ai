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


from hugegraph_llm.utils.config import Config
from hugegraph_llm.utils.constants import Constants
from pyhugegraph.client import PyHugeClient


def generate_new_relationships(nodes_schemas_data, relationships_data):
    label_id = {}
    i = 1
    old_label = []
    for item in nodes_schemas_data:
        label = item["label"]
        if label in old_label:
            continue
        label_id[label] = i
        i += 1
        old_label.append(label)
    new_relationships_data = []
    for relationship in relationships_data:
        start = relationship["start"]
        end = relationship["end"]
        relationships_type = relationship["type"]
        properties = relationship["properties"]
        new_start = []
        new_end = []
        for key, value in label_id.items():
            for key1, value1 in start.items():
                if key1 == key:
                    new_start = f"{value}" + ":" + f"{value1}"
            for key1, value1 in end.items():
                if key1 == key:
                    new_end = f"{value}" + ":" + f"{value1}"
        relationships_data = {}
        relationships_data["start"] = new_start
        relationships_data["end"] = new_end
        relationships_data["type"] = relationships_type
        relationships_data["properties"] = properties
        new_relationships_data.append(relationships_data)
    return new_relationships_data


def generate_schema_properties(data):
    schema_properties_statements = []
    if len(data) == 3:
        for item in data:
            properties = item["properties"]
            for key, value in properties.items():
                if value == "int":
                    schema_properties_statements.append(
                        f"schema.propertyKey('{key}').asInt().ifNotExist().create()"
                    )
                elif value == "text":
                    schema_properties_statements.append(
                        f"schema.propertyKey('{key}').asText().ifNotExist().create()"
                    )
    else:
        for item in data:
            properties = item["properties"]
            for key, value in properties.items():
                if value == "int":
                    schema_properties_statements.append(
                        f"schema.propertyKey('{key}').asInt().ifNotExist().create()"
                    )
                elif value == "text":
                    schema_properties_statements.append(
                        f"schema.propertyKey('{key}').asText().ifNotExist().create()"
                    )
    return schema_properties_statements


def generate_schema_nodes(data):
    schema_nodes_statements = []
    for item in data:
        label = item["label"]
        primary_key = item["primary_key"]
        properties = item["properties"]
        schema_statement = f"schema.vertexLabel('{label}').properties("
        schema_statement += ", ".join(f"'{prop}'" for prop in properties.keys())
        schema_statement += ").nullableKeys("
        schema_statement += ", ".join(
            f"'{prop}'" for prop in properties.keys() if prop != primary_key
        )
        schema_statement += (
            f").usePrimaryKeyId().primaryKeys('{primary_key}').ifNotExist().create()"
        )
        schema_nodes_statements.append(schema_statement)
    return schema_nodes_statements


def generate_schema_relationships(data):
    schema_relationships_statements = []
    for item in data:
        start = item["start"]
        end = item["end"]
        schema_relationships_type = item["type"]
        properties = item["properties"]
        schema_statement = (
            f"schema.edgeLabel('{schema_relationships_type}')"
            f".sourceLabel('{start}').targetLabel('{end}').properties("
        )
        schema_statement += ", ".join(f"'{prop}'" for prop in properties.keys())
        schema_statement += ").nullableKeys("
        schema_statement += ", ".join(f"'{prop}'" for prop in properties.keys())
        schema_statement += ").ifNotExist().create()"
        schema_relationships_statements.append(schema_statement)
    return schema_relationships_statements


def generate_nodes(data):
    nodes = []
    for item in data:
        label = item["label"]
        properties = item["properties"]
        nodes.append(f"g.addVertex('{label}', {properties})")
    return nodes


def generate_relationships(data):
    relationships = []
    for item in data:
        start = item["start"]
        end = item["end"]
        types = item["type"]
        properties = item["properties"]
        relationships.append(f"g.addEdge('{types}', '{start}', '{end}', {properties})")
    return relationships


class CommitDataToKg:
    def __init__(self):
        config = Config(section=Constants.HUGEGRAPH_CONFIG)
        self.client = PyHugeClient(
            config.get_graph_ip(),
            config.get_graph_port(),
            config.get_graph_user(),
            config.get_graph_pwd(),
            config.get_graph_name(),
        )
        self.schema = self.client.schema()

    def run(self, data: dict):
        # # If you are using a http proxy, you can run the following code to unset http proxy
        # os.environ.pop("http_proxy")
        # os.environ.pop("https_proxy")
        nodes = data["nodes"]
        relationships = data["relationships"]
        nodes_schemas = data["nodes_schemas"]
        relationships_schemas = data["relationships_schemas"]
        # properties schema
        schema_nodes_properties = generate_schema_properties(nodes_schemas)
        schema_relationships_properties = generate_schema_properties(
            relationships_schemas
        )
        for schema_nodes_property in schema_nodes_properties:
            exec(schema_nodes_property)

        for schema_relationships_property in schema_relationships_properties:
            exec(schema_relationships_property)

        # nodes schema
        schema_nodes = generate_schema_nodes(nodes_schemas)
        for schema_node in schema_nodes:
            exec(schema_node)

        # relationships schema
        schema_relationships = generate_schema_relationships(relationships_schemas)
        for schema_relationship in schema_relationships:
            exec(schema_relationship)

        # nodes
        nodes = generate_nodes(nodes)
        for node in nodes:
            exec(node)

        # relationships
        new_relationships = generate_new_relationships(nodes_schemas, relationships)
        relationships_schemas = generate_relationships(new_relationships)
        for relationship in relationships_schemas:
            exec(relationship)
