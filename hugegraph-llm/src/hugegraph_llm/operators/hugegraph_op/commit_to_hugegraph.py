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
            settings.graph_space,
        )
        self.schema = self.client.schema()

    def run(self, data: dict) -> Dict[str, Any]:
        schema = data.get("schema")
        vertices = data.get("vertices", [])
        edges = data.get("edges", [])

        if not schema:
            # TODO: ensure the function works correctly (update the logic later)
            self.schema_free_mode(data.get("triples", []))
            log.warning("Using schema_free mode, could try schema_define mode for better effect!")
        else:
            self.init_schema_if_need(schema)
            self.load_into_graph(vertices, edges, schema)
        return data

    def load_into_graph(self, vertices, edges, schema):
        key_map = {}
        for vlabel in schema["vertexlabels"]:
            key_map[vlabel["name"]] = vlabel
        for vertex in vertices:
            label = vertex["label"]
            properties = vertex["properties"]
            if label not in key_map:
                log.warning("Vertex label %s not found in schema, ignored!", label)
                continue
            primary_keys = key_map[label]["primary_keys"]
            if len(primary_keys) == 1:
                # Single primary key
                if primary_keys[0] not in properties:
                    log.warning("Vertex %s missing primary key %s, ignored!", vertex, primary_keys[0])
                    continue
            elif len(primary_keys) > 1:
                # Composite primary key
                for pk in primary_keys:
                    if pk not in properties:
                        properties[pk] = ""
            nullable_keys = key_map[label]["nullable_keys"] if "nullable_keys" in key_map[label] else []
            nonnull_keys = [
                key for key in key_map[label]["properties"] if key not in nullable_keys
            ]
            for key in nonnull_keys:
                if key not in properties:
                    log.warning("Vertex %s missing property %s, set to null!", vertex, key)
                    properties[key] = "null"
            try:
                # TODO: we could try batch add vertices first, setback to single-mode if failed
                vid = self.client.graph().addVertex(label, properties).id
                vertex["id"] = vid
            except NotFoundError as e:
                log.error(e)
            except CreateError as e:
                log.error("Error on creating vertex: %s, %s", vertex, e)

        for edge in edges:
            start = edge["outV"]
            end = edge["inV"]
            label = edge["label"]
            properties = edge["properties"]
            try:
                # TODO: we could try batch add edges first, setback to single-mode if failed
                self.client.graph().addEdge(label, start, end, properties)
            except NotFoundError as e:
                log.error(e)
            except CreateError as e:
                log.error("Error on creating edge: %s, %s", edge, e)

    def init_schema_if_need(self, schema: object):
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
        self.schema.vertexLabel("vertex").useCustomizeStringId().properties("name").ifNotExist().create()
        self.schema.edgeLabel("edge").sourceLabel("vertex").targetLabel("vertex").properties(
            "name"
        ).ifNotExist().create()

        self.schema.indexLabel("vertexByName").onV("vertex").by("name").secondary().ifNotExist().create()
        self.schema.indexLabel("edgeByName").onE("edge").by("name").secondary().ifNotExist().create()

        for item in data:
            s, p, o = (element.strip() for element in item)
            s_id = self.client.graph().addVertex("vertex", {"name": s}, id=s).id
            t_id = self.client.graph().addVertex("vertex", {"name": o}, id=o).id
            self.client.graph().addEdge("edge", s_id, t_id, {"name": p})
