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

from pyhugegraph.client import PyHugeClient
from hugegraph_llm.config import settings


def run_gremlin_query(query, format=False):
    res = get_hg_client().gremlin().exec(query)
    return json.dumps(res, indent=4, ensure_ascii=False) if format else res


def get_hg_client():
    return PyHugeClient(
        settings.graph_ip,
        settings.graph_port,
        settings.graph_name,
        settings.graph_user,
        settings.graph_pwd,
        settings.graph_space,
    )


def init_hg_test_data():
    client = get_hg_client()
    client.graphs().clear_graph_all_data()
    schema = client.schema()
    schema.propertyKey("name").asText().ifNotExist().create()
    schema.propertyKey("birthDate").asText().ifNotExist().create()
    schema.vertexLabel("Person").properties("name", "birthDate").useCustomizeStringId().ifNotExist().create()
    schema.vertexLabel("Movie").properties("name").useCustomizeStringId().ifNotExist().create()
    schema.edgeLabel("ActedIn").sourceLabel("Person").targetLabel("Movie").ifNotExist().create()

    schema.indexLabel("PersonByName").onV("Person").by("name").secondary().ifNotExist().create()
    schema.indexLabel("MovieByName").onV("Movie").by("name").secondary().ifNotExist().create()

    graph = client.graph()
    graph.addVertex("Person", {"name": "Al Pacino", "birthDate": "1940-04-25"}, id="Al Pacino")
    graph.addVertex(
        "Person",
        {"name": "Robert De Niro", "birthDate": "1943-08-17"},
        id="Robert De Niro",
    )
    graph.addVertex("Movie", {"name": "The Godfather"}, id="The Godfather")
    graph.addVertex("Movie", {"name": "The Godfather Part II"}, id="The Godfather Part II")
    graph.addVertex(
        "Movie",
        {"name": "The Godfather Coda The Death of Michael Corleone"},
        id="The Godfather Coda The Death of Michael Corleone",
    )

    graph.addEdge("ActedIn", "Al Pacino", "The Godfather", {})
    graph.addEdge("ActedIn", "Al Pacino", "The Godfather Part II", {})
    graph.addEdge("ActedIn", "Al Pacino", "The Godfather Coda The Death of Michael Corleone", {})
    graph.addEdge("ActedIn", "Robert De Niro", "The Godfather Part II", {})
    schema.getSchema()
    graph.close()
    return {
        "vertex": ["Person", "Movie"],
        "edge": ["ActedIn"],
        "property": ["name", "birthDate"],
        "index": ["PersonByName", "MovieByName"],
    }


def clean_hg_data():
    client = get_hg_client()
    client.graphs().clear_graph_all_data()
