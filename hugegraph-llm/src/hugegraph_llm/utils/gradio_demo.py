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
import os

import gradio as gr
import uvicorn
from fastapi import FastAPI

from hugegraph_llm.llms.init_llm import LLMs
from hugegraph_llm.operators.graph_rag_task import GraphRAG
from hugegraph_llm.operators.kg_construction_task import KgBuilder
from hugegraph_llm.utils.config import Config
from hugegraph_llm.utils.constants import Constants
from pyhugegraph.client import PyHugeClient


def init_hg_test_data():
    client = get_hg_client()
    client.graphs().clear_graph_all_data()
    schema = client.schema()
    schema.propertyKey("name").asText().ifNotExist().create()
    schema.propertyKey("birthDate").asText().ifNotExist().create()
    schema.vertexLabel("Person").properties(
        "name", "birthDate"
    ).useCustomizeStringId().ifNotExist().create()
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


def graph_rag(text):
    res = (
        GraphRAG()
        .extract_keyword(text=text)
        .query_graph_for_rag()
        .synthesize_answer()
        .run(verbose=True)
    )
    return res


def build_kg(text, schema, disambiguate_word_sense, commit_to_hugegraph):
    builder = KgBuilder(LLMs().get_llm())
    if schema:
        try:
            schema = json.loads(schema.strip())
            builder.import_schema(from_user_defined=schema)
        except json.JSONDecodeError as e:
            print(e)
            builder.import_schema(from_hugegraph=schema)
    else:
        return "ERROR: please input schema."
    builder.extract_triples(text)
    if disambiguate_word_sense == "true":
        builder.disambiguate_word_sense()
    if commit_to_hugegraph == "true":
        builder.commit_to_hugegraph()
    return builder.run()


def run_gremlin_query(query):
    res = get_hg_client().gremlin().exec(query)
    return res


def get_hg_client():
    config = Config(section=Constants.HUGEGRAPH_CONFIG)
    return PyHugeClient(
        config.get_graph_ip(),
        config.get_graph_port(),
        config.get_graph_name(),
        config.get_graph_user(),
        config.get_graph_pwd(),
    )


def init_config(
    ip, port, user, pwd, graph, type, api_key, secret_key, llm_url, model_name, max_token
):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_file = os.path.join(root_dir, "config", "config.ini")

    config = Config(config_file=config_file, section="hugegraph")
    config.update_config({"ip": ip, "port": port, "user": user, "pwd": pwd, "graph": graph})

    config = Config(config_file=config_file, section="llm")
    config.update_config(
        {
            "type": type,
            "api_key": api_key,
            "secret_key": secret_key,
            "llm_url": llm_url,
            "model_name": model_name,
            "max_token": max_token,
        }
    )
    with open(config_file, "r", encoding="utf-8") as file:
        content = file.read()
    return content


with gr.Blocks() as hugegraph_llm:
    gr.Markdown(
        """# HugeGraph LLM Demo
    1. Set up the HugeGraph server."""
    )
    with gr.Row():
        inp = [
            gr.Textbox(value="127.0.0.1", label="ip"),
            gr.Textbox(value="8080", label="port"),
            gr.Textbox(value="admin", label="user"),
            gr.Textbox(value="admin", label="pwd"),
            gr.Textbox(value="hugegraph", label="graph"),
        ]
    gr.Markdown("2. Set up the LLM.")
    with gr.Row():
        inp2 = [
            gr.Textbox(value="ernie", label="type"),
            gr.Textbox(value="", label="api_key"),
            gr.Textbox(value="", label="secret_key"),
            gr.Textbox(
                value="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/"
                "chat/completions_pro?access_token=",
                label="llm_url",
            ),
            gr.Textbox(value="wenxin", label="model_name"),
            gr.Textbox(value="4000", label="max_token"),
        ]
    with gr.Row():
        out = gr.Textbox(label="Output")
    btn = gr.Button("Initialize configs")
    btn.click(fn=init_config, inputs=inp + inp2, outputs=out)  # pylint: disable=no-member

    gr.Markdown(
        """## 1. build knowledge graph
    - Text: The input text.
    - Schema: Accepts two types of text as below:
        - User-defined JSON format Schema. 
        - Specify the name of the HugeGraph graph instance, and it will 
        automatically extract the schema of the graph.
    - Disambiguate word sense: Whether to perform word sense disambiguation.
    - Commit to hugegraph: Whether to commit the constructed knowledge graph to the 
    HugeGraph server.
    """
    )
    TEXT = (
        "Meet Sarah, a 30-year-old attorney, and her roommate, James, whom she's shared a home with"
        " since 2010. James, in his professional life, works as a journalist. Additionally, Sarah"
        " is the proud owner of the website www.sarahsplace.com, while James manages his own"
        " webpage, though the specific URL is not mentioned here. These two individuals, Sarah and"
        " James, have not only forged a strong personal bond as roommates but have also carved out"
        " their distinctive digital presence through their respective webpages, showcasing their"
        " varied interests and experiences."
    )

    SCHEMA = """{
        "vertices": [
            {"vertex_label": "person", "properties": ["name", "age", "occupation"]},
            {"vertex_label": "webpage", "properties": ["name", "url"]}
        ],
        "edges": [
            {
                "edge_label": "roommate",
                "source_vertex_label": "person",
                "target_vertex_label": "person",
                "properties": {}
            }
        ]
    }
    """

    with gr.Row():
        inp = [
            gr.Textbox(value=TEXT, label="Text"),
            gr.Textbox(value=SCHEMA, label="Schema"),
            gr.Textbox(value="false", label="Disambiguate word sense"),
            gr.Textbox(value="false", label="Commit to hugegraph"),
        ]
    with gr.Row():
        out = gr.Textbox(label="Output")
    btn = gr.Button("Build knowledge graph")
    btn.click(fn=build_kg, inputs=inp, outputs=out)  # pylint: disable=no-member

    gr.Markdown("""## 2. Retrieval augmented generation by hugegraph""")
    with gr.Row():
        inp = gr.Textbox(value="Tell me about Al Pacino.", label="Question")
    with gr.Row():
        out = gr.Textbox(label="Answer")
    btn = gr.Button("Retrieval augmented generation")
    btn.click(fn=graph_rag, inputs=inp, outputs=out)  # pylint: disable=no-member

    gr.Markdown("""## 3. Others """)
    with gr.Row():
        inp = []
        out = gr.Textbox(label="Output")
    btn = gr.Button("Initialize HugeGraph test data")
    btn.click(fn=init_hg_test_data, inputs=inp, outputs=out)  # pylint: disable=no-member

    with gr.Row():
        inp = gr.Textbox(value="g.V().limit(10)", label="Gremlin query")
        out = gr.Textbox(label="Output")
    btn = gr.Button("Run gremlin query on HugeGraph")
    btn.click(fn=run_gremlin_query, inputs=inp, outputs=out)  # pylint: disable=no-member

if __name__ == '__main__':
    app = FastAPI()
    app = gr.mount_gradio_app(app, hugegraph_llm, path="/")
    uvicorn.run(app, host="0.0.0.0", port=8001)
