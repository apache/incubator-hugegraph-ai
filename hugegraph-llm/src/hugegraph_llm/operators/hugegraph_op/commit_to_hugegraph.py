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


from typing import Dict, Any

from hugegraph_llm.config import settings
from hugegraph_llm.utils.log import log
from pyhugegraph.client import PyHugeClient
from pyhugegraph.utils.exceptions import NotFoundError, CreateError


class CommitToKg:
    def __init__(self):
        self.client = PyHugeClient(
            settings.graph_ip,
            settings.graph_port,
            settings.graph_name,
            settings.graph_user,
            settings.graph_pwd,
        )
        self.schema = self.client.schema()

    def run(self, data: dict) -> Dict[str, Any]:
        if "schema" not in data:
            self.schema_free_mode(data["triples"])
        else:
            schema = data["schema"]
            vertices = data["vertices"]
            edges = data["edges"]
            self.init_schema(schema)
            self.init_graph(vertices, edges, schema)
        return data

    def init_graph(self, vertices, edges, schema):
        key_map = {}
        for vertex in schema["vertexlabels"]:
            key_map[vertex["name"]] = vertex
        for vertex in vertices:
            label = vertex["label"]
            properties = vertex["properties"]
            for pk in key_map[label]["primary_keys"]:
                if pk not in properties:
                    properties[pk] = "NULL"
            for uk in key_map[label]["nullable_keys"]:
                if uk not in properties:
                    properties[uk] = "NULL"
            try:
                vid = self.client.graph().addVertex(label, properties).id
                vertex["id"] = vid
            except NotFoundError as e:
                print(e)
        for edge in edges:
            start = edge["outV"]
            end = edge["inV"]
            label = edge["label"]
            properties = edge["properties"]
            try:
                self.client.graph().addEdge(label, start, end, properties)
            except NotFoundError as e:
                print(e)
            except CreateError as e:
                log.error("Error on creating edge: %s", str(edge))
                print(e)

    def init_schema(self, schema):
        vertices = schema["vertexlabels"]
        edges = schema["edgelabels"]

        for vertex in vertices:
            vertex_label = vertex["name"]
            properties = vertex["properties"]
            nullable_keys = vertex["nullable_keys"]
            primary_keys = vertex["primary_keys"]
            for prop in properties:
                self.schema.propertyKey(prop).asText().ifNotExist().create()
            self.schema.vertexLabel(vertex_label).properties(*properties).nullableKeys(
                *nullable_keys
            ).usePrimaryKeyId().primaryKeys(*primary_keys).ifNotExist().create()
        for edge in edges:
            edge_label = edge["name"]
            source_vertex_label = edge["source_label"]
            target_vertex_label = edge["target_label"]
            properties = edge["properties"]
            for prop in properties:
                self.schema.propertyKey(prop).asText().ifNotExist().create()
            self.schema.edgeLabel(edge_label).sourceLabel(source_vertex_label).targetLabel(
                target_vertex_label
            ).properties(*properties).nullableKeys(*properties).ifNotExist().create()

    def schema_free_mode(self, data):
        self.schema.propertyKey("name").asText().ifNotExist().create()
        self.schema.vertexLabel("vertex").useCustomizeStringId().properties(
            "name"
        ).ifNotExist().create()
        self.schema.edgeLabel("edge").sourceLabel("vertex").targetLabel("vertex").properties(
            "name"
        ).ifNotExist().create()

        self.schema.indexLabel("vertexByName").onV("vertex").by(
            "name"
        ).secondary().ifNotExist().create()
        self.schema.indexLabel("edgeByName").onE("edge").by(
            "name"
        ).secondary().ifNotExist().create()

        for item in data:
            s, p, o = (element.strip() for element in item)
            s_id = self.client.graph().addVertex("vertex", {"name": s}, id=s).id
            t_id = self.client.graph().addVertex("vertex", {"name": o}, id=o).id
            self.client.graph().addEdge("edge", s_id, t_id, {"name": p})
