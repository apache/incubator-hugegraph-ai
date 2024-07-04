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

import requests
import uvicorn
import docx
import gradio as gr
from fastapi import FastAPI

from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.operators.graph_rag_task import GraphRAG
from hugegraph_llm.operators.kg_construction_task import KgBuilder
from hugegraph_llm.config import settings
from hugegraph_llm.utils.hugegraph_utils import (
    init_hg_test_data,
    get_hg_client,
    run_gremlin_query
)
from hugegraph_llm.utils.log import log


def graph_rag(text, vector_search: str):
    searcher = GraphRAG().extract_keyword(text=text).query_graph_for_rag()
    if vector_search == "true":
        searcher.query_vector_index_for_rag()
    return searcher.merge_dedup_rerank().synthesize_answer().run(verbose=True)


def build_kg(file, schema, disambiguate_word_sense, commit_to_hugegraph, build_vector_index):
    if str(file).endswith(".txt"):
        with open(str(file), "r", encoding="utf-8") as f:
            text = f.read()
    elif str(file).endswith(".docx"):
        text = ""
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text
            text += "\n"
    elif str(file).endswith(".pdf"):
        raise Exception("ERROR: PDF will be supported later!")
    else:
        return "ERROR: please input txt or docx file."
    builder = KgBuilder(LLMs().get_llm(), Embeddings().get_embedding())
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
        client = get_hg_client()
        client.graphs().clear_graph_all_data()
        builder.commit_to_hugegraph()
    if build_vector_index == "true":
        builder.do_triples_embedding()
        builder.build_vector_index()
    return builder.run()


if __name__ == "__main__":
    app = FastAPI()
    with gr.Blocks() as hugegraph_llm:
        gr.Markdown(
            """# HugeGraph LLM RAG Demo
        1. Set up the HugeGraph server."""
        )
        with gr.Row():
            graph_config_input = [
                gr.Textbox(value=settings.graph_ip, label="ip"),
                gr.Textbox(value=str(settings.graph_port), label="port"),
                gr.Textbox(value=settings.graph_user, label="user"),
                gr.Textbox(value=settings.graph_pwd, label="pwd"),
                gr.Textbox(value=settings.graph_name, label="graph")
            ]
        graph_config_button = gr.Button("apply configuration")


        def test_api_connection(url, method="GET", ak=None, sk=None, headers=None, body=None):
            # TODO: use fastapi.request / starlette instead? (Also add a try-catch here)
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=5)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=body, timeout=5)
            else:
                log.error(f"Unsupported method: {method}")
                return


            if response.status_code >= 200 and response.status_code < 300:
                log.info("Connection successful. Configured finished.")
                gr.Info("Connection successful. Configured finished.")
            else:
                log.error(f"Connection failed with status code: {response.status_code}")
                gr.Error(f"Connection failed with status code: {response.status_code}")


        def apply_graph_configuration(ip, port, user, pwd, name):
            settings.graph_ip = ip
            settings.graph_port = int(port)
            settings.graph_user = user
            settings.graph_pwd = pwd
            settings.graph_name = name
            test_url = f"http://{settings.graph_ip}:{settings.graph_port}/graphs/{settings.graph_name}/schema"
            test_api_connection(test_url)


        graph_config_button.click(apply_graph_configuration, inputs=graph_config_input)  # pylint: disable=no-member

        gr.Markdown("2. Set up the LLM.")
        llm_dropdown = gr.Dropdown(
            choices=["openai", "qianfan_wenxin", "ollama"],
            value=settings.llm_type,
            label="LLM"
        )


        @gr.render(inputs=[llm_dropdown])
        def llm_settings(llm_type):
            settings.llm_type = llm_type
            if llm_type == "openai":
                with gr.Row():
                    llm_config_input = [
                        gr.Textbox(value=settings.openai_api_key, label="api_key"),
                        gr.Textbox(value=settings.openai_api_base, label="api_base"),
                        gr.Textbox(value=settings.openai_language_model, label="model_name"),
                        gr.Textbox(value=str(settings.openai_max_tokens), label="max_token"),
                    ]
            elif llm_type == "qianfan_wenxin":
                with gr.Row():
                    llm_config_input = [
                        gr.Textbox(value=settings.qianfan_api_key, label="api_key"),
                        gr.Textbox(value=settings.qianfan_secret_key, label="secret_key"),
                        gr.Textbox(value=settings.qianfan_chat_url, label="chat_url"),
                        gr.Textbox(value=settings.qianfan_chat_name, label="model_name")
                    ]
                log.debug(llm_config_input)
            elif llm_type == "ollama":
                with gr.Row():
                    llm_config_input = [
                        gr.Textbox(value=settings.ollama_host, label="host"),
                        gr.Textbox(value=str(settings.ollama_port), label="port"),
                        gr.Textbox(value=settings.ollama_language_model, label="model_name"),
                        gr.Textbox(value="", visible=False)
                    ]
            else:
                llm_config_input = []
            llm_config_button = gr.Button("apply configuration")

            def apply_llm_configuration(arg1, arg2, arg3, arg4):
                llm_type = settings.llm_type

                if llm_type == "openai":
                    settings.openai_api_key = arg1
                    settings.openai_api_base = arg2
                    settings.openai_language_model = arg3
                    settings.openai_max_tokens = arg4
                    test_url = "https://api.openai.com/v1/models"
                    headers = {"Authorization": f"Bearer {arg1}"}
                    test_api_connection(test_url, headers=headers, ak=arg1)
                elif llm_type == "qianfan_wenxin":
                    settings.qianfan_api_key = arg1
                    settings.qianfan_secret_key = arg2
                    settings.qianfan_chat_url = arg3
                    settings.qianfan_chat_name = arg4
                    log.debug("####")
                    log.debug(LLMs().get_llm().base_url)
                    #test_url = "https://aip.baidubce.com/oauth/2.0/token"  # POST
                elif llm_type == "ollama":
                    settings.ollama_host = arg1
                    settings.ollama_port = int(arg2)
                    settings.ollama_language_model = arg3
                gr.Info("configured!")

            llm_config_button.click(apply_llm_configuration, inputs=llm_config_input)  # pylint: disable=no-member


        gr.Markdown("3. Set up the Embedding.")
        embedding_dropdown = gr.Dropdown(
            choices=["openai", "ollama", "qianfan_wenxin"],
            value=settings.embedding_type,
            label="Embedding"
        )


        @gr.render(inputs=[embedding_dropdown])
        def embedding_settings(embedding_type):
            settings.embedding_type = embedding_type
            if embedding_type == "openai":
                with gr.Row():
                    embedding_config_input = [
                        gr.Textbox(value=settings.openai_api_key, label="api_key"),
                        gr.Textbox(value=settings.openai_api_base, label="api_base"),
                        gr.Textbox(value=settings.openai_embedding_model, label="model_name")
                    ]
            elif embedding_type == "ollama":
                with gr.Row():
                    embedding_config_input = [
                        gr.Textbox(value=settings.ollama_host, label="host"),
                        gr.Textbox(value=str(settings.ollama_port), label="port"),
                        gr.Textbox(value=settings.ollama_embedding_model, label="model_name"),
                    ]
            elif embedding_type == "qianfan_wenxin":
                with gr.Row():
                    embedding_config_input = [
                        gr.Textbox(value=settings.qianfan_access_token, label="access_token (from AK+SK)"),
                        gr.Textbox(value=settings.qianfan_embed_url, label="api_base"),
                        gr.Textbox(value=settings.qianfan_embedding_name, label="embed_name")
                    ]
            else:
                embedding_config_input = []
            embedding_config_button = gr.Button("apply configuration")

            def apply_embedding_configuration(arg1, arg2, arg3):
                embedding_type = settings.embedding_type
                if embedding_type == "openai":
                    settings.openai_api_key = arg1
                    settings.openai_api_base = arg2
                    settings.openai_embedding_model = arg3
                    test_url = "https://api.openai.com/v1/models"
                    headers = {"Authorization": f"Bearer {arg1}"}
                    test_api_connection(test_url, headers=headers, ak=arg1)
                elif embedding_type == "ollama":
                    settings.ollama_host = arg1
                    settings.ollama_port = int(arg2)
                    settings.ollama_embedding_model = arg3
                elif embedding_type == "qianfan_wenxin":
                    settings.qianfan_access_token = arg1
                    settings.qianfan_embed_url = arg2

                gr.Info("configured!")

            embedding_config_button.click(apply_embedding_configuration,  # pylint: disable=no-member
                                          inputs=embedding_config_input)


        gr.Markdown(
            """## 1. build knowledge graph
- Document: Input document file which should be TXT or DOCX.
- Schema: Accepts two types of text as below:
    - User-defined JSON format Schema.
    - Specify the name of the HugeGraph graph instance, and it will
    automatically extract the schema of the graph.
- Disambiguate word sense: Whether to perform word sense disambiguation.
- Commit to hugegraph: Whether to commit the constructed knowledge graph to the HugeGraph server.
- Build vector index: Whether to build vector index to help retrieving.
"""
        )

        SCHEMA = """{
    "vertices": [
        {"vertex_label": "entity", "properties": []}
    ],
    "edges": [
        {
            "edge_label": "relation",
            "source_vertex_label": "entity",
            "target_vertex_label": "entity",
            "properties": {}
        }
    ]
}"""

        with gr.Row():
            input_file = gr.File(value=None, label="Document")
            input_schema = gr.Textbox(value=SCHEMA, label="Schema")
            with gr.Column():
                disambiguate_word_sense_radio = gr.Radio(choices=["true", "false"], value="false",
                                                         label="Disambiguate word sense")
                commit_to_hugegraph_radio = gr.Radio(choices=["true", "false"], value="false",
                                                     label="Commit to hugegraph")
                build_vector_index_radio = gr.Radio(choices=["true", "false"], value="false",
                                                    label="Build vector index")
        with gr.Row():
            out = gr.Textbox(label="Output")
        btn = gr.Button("Build knowledge graph")
        btn.click(  # pylint: disable=no-member
            fn=build_kg,
            inputs=[input_file, input_schema, disambiguate_word_sense_radio,
                    commit_to_hugegraph_radio, build_vector_index_radio],
            outputs=out
        )  # pylint: disable=no-member

        gr.Markdown("""## 2. Retrieval augmented generation by hugegraph""")
        with gr.Row():
            with gr.Column(scale=2):
                inp = gr.Textbox(value="Tell me about Al Pacino.", label="Question")
                out = gr.Textbox(label="Answer")
            with gr.Column(scale=1):
                vector_search_radio = gr.Radio(choices=["true", "false"], value="false",
                                               label="Vector search")
                btn = gr.Button("Retrieval augmented generation")
        btn.click(fn=graph_rag, inputs=[inp, vector_search_radio], outputs=out)  # pylint: disable=no-member

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
    app = gr.mount_gradio_app(app, hugegraph_llm, path="/")
    # Note: set reload to False in production environment
    uvicorn.run(app, host="0.0.0.0", port=8001)
    # TODO: we can't use reload now due to the config 'app' of uvicorn.run
    # uvicorn.run("rag_web_demo:app", host="0.0.0.0", port=8001, reload=True)