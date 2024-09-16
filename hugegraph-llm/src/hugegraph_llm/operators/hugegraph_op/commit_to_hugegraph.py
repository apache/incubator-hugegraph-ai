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

        if not vertices and not edges:
            log.critical("(Loading) Both vertices and edges are empty. Please check the input data again.")
            raise ValueError("Both vertices and edges input are empty.")

        if not schema:
            # TODO: ensure the function works correctly (update the logic later)
            self.schema_free_mode(data.get("triples", []))
            log.warning("Using schema_free mode, could try schema_define mode for better effect!")
        else:
            self.init_schema_if_need(schema)
            self.load_into_graph(vertices, edges, schema)
        return data

    def load_into_graph(self, vertices, edges, schema):
        vertex_label_map = {v_label["name"]: v_label for v_label in schema["vertexlabels"]}
        edge_label_map = {e_label["name"]: e_label for e_label in schema["edgelabels"]}

        for vertex in vertices:
            input_label = vertex["label"]
            # 1. ensure the input_label in the graph schema
            if input_label not in vertex_label_map:
                log.critical("(Input) VertexLabel %s not found in schema, skip & need check it!", input_label)
                continue

            input_properties = vertex["properties"]
            vertex_label = vertex_label_map[input_label]
            primary_keys = vertex_label["primary_keys"]
            nullable_keys = vertex_label.get("nullable_keys", [])
            non_null_keys = [key for key in vertex_label["properties"] if key not in nullable_keys]

            # 2. Handle primary-keys mode vertex
            for pk in primary_keys:
                if not input_properties.get(pk):
                    if len(primary_keys) == 1:
                        log.error("Primary-key '%s' missing in vertex %s, skip it & need check it again", pk, vertex)
                        continue
                    input_properties[pk] = "null" # FIXME: handle bool/number/date type
                    log.warning("Primary-key '%s' missing in vertex %s, mark empty & need check it again!", pk, vertex)

            # 3. Ensure all non-nullable props are set
            for key in non_null_keys:
                if key not in input_properties:
                    input_properties[key] = "" # FIXME: handle bool/number/date type
                    log.warning("Property '%s' missing in vertex %s, set to '' for now", key, vertex)
            try:
                # TODO: we could try batch add vertices first, setback to single-mode if failed
                vid = self.client.graph().addVertex(input_label, input_properties).id
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

            if label not in edge_label_map:
                log.critical("(Input) EdgeLabel %s not found in schema, skip & need check it!", label)
                continue
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
