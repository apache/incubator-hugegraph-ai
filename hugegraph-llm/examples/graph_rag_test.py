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


import os

from hugegraph_llm.operators.graph_rag_task import GraphRAG
from pyhugegraph.client import PyHugeClient


def prepare_data():
    client = PyHugeClient("127.0.0.1", 8080, "hugegraph", "admin", "admin")
    schema = client.schema()
    schema.propertyKey("name").asText().ifNotExist().create()
    schema.propertyKey("birthDate").asText().ifNotExist().create()
    schema.vertexLabel("Person").properties(
        "name", "birthDate"
    ).useCustomizeStringId().ifNotExist().create()
    schema.vertexLabel("Movie").properties("name").useCustomizeStringId().ifNotExist().create()
    schema.indexLabel("PersonByName").onV("Person").by("name").secondary().ifNotExist().create()
    schema.indexLabel("MovieByName").onV("Movie").by("name").secondary().ifNotExist().create()
    schema.edgeLabel("ActedIn").sourceLabel("Person").targetLabel("Movie").ifNotExist().create()

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

    graph.close()


if __name__ == "__main__":
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ["OPENAI_API_KEY"] = ""

    # prepare_data()

    graph_rag = GraphRAG()

    # configure operator with context dict
    context = {
        # hugegraph client
        "ip": "localhost",  # default to "localhost" if not set
        "port": 18080,  # default to 8080 if not set
        "user": "admin",  # default to "admin" if not set
        "pwd": "admin",  # default to "admin" if not set
        "graph": "hugegraph",  # default to "hugegraph" if not set
        # query question
        "query": "Tell me about Al Pacino.",  # must be set
        # keywords extraction
        "max_keywords": 5,  # default to 5 if not set
        "language": "english",  # default to "english" if not set
        # graph rag query
        "prop_to_match": "name",  # default to None if not set
        "max_deep": 2,  # default to 2 if not set
        "max_items": 30,  # default to 30 if not set
        # print intermediate processes result
        "verbose": True,  # default to False if not set
    }
    result = graph_rag.extract_keyword().query_graph_for_rag().synthesize_answer().run(**context)
    print(f"Query:\n- {context['query']}")
    print(f"Answer:\n- {result['answer']}")

    print("--------------------------------------------------------")

    # configure operator with parameters
    graph_client = PyHugeClient("127.0.0.1", 18080, "hugegraph", "admin", "admin")
    result = (
        graph_rag.extract_keyword(
            text="Tell me about Al Pacino.",
            max_keywords=5,  # default to 5 if not set
            language="english",  # default to "english" if not set
        )
        .query_graph_for_rag(
            graph_client=graph_client,
            max_deep=2,  # default to 2 if not set
            max_items=30,  # default to 30 if not set
            prop_to_match=None,  # default to None if not set
        )
        .synthesize_answer()
        .run(verbose=True)
    )
    print("Query:\n- Tell me about Al Pacino.")
    print(f"Answer:\n- {result['answer']}")
