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
import argparse
import os
from typing import Optional

import requests
import uvicorn
import docx
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth

from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.operators.graph_rag_task import GraphRAG
from hugegraph_llm.operators.kg_construction_task import KgBuilder
from hugegraph_llm.config import settings, resource_path
from hugegraph_llm.operators.llm_op.property_graph_extract import SCHEMA_EXAMPLE_PROMPT
from hugegraph_llm.utils.hugegraph_utils import (
    init_hg_test_data,
    run_gremlin_query,
    clean_hg_data
)
from hugegraph_llm.utils.log import log
from hugegraph_llm.utils.hugegraph_utils import get_hg_client
from hugegraph_llm.utils.vector_index_utils import clean_vector_index


def convert_bool_str(string):
    if string == "true":
        return True
    if string == "false":
        return False
    raise gr.Error(f"Invalid boolean string: {string}")


# TODO: enhance/distinguish the "graph_rag" name to avoid confusion
def graph_rag(text: str, raw_answer: bool, vector_only_answer: bool,
              graph_only_answer: bool, graph_vector_answer: bool):
    vector_search = vector_only_answer or graph_vector_answer
    graph_search = graph_only_answer or graph_vector_answer

    if raw_answer == False and not vector_search and not graph_search:
        gr.Warning("Please select at least one generate mode.")
        return "", "", "", ""
    searcher = GraphRAG()
    if vector_search:
        searcher.query_vector_index_for_rag()
    if graph_search:
        searcher.extract_keyword().match_keyword_to_id().query_graph_for_rag()
    searcher.merge_dedup_rerank().synthesize_answer(
        raw_answer=raw_answer,
        vector_only_answer=vector_only_answer,
        graph_only_answer=graph_only_answer,
        graph_vector_answer=graph_vector_answer
    ).run(verbose=True, query=text)

    try:
        context = searcher.run(verbose=True, query=text)
        return (
            context.get("raw_answer", ""),
            context.get("vector_only_answer", ""),
            context.get("graph_only_answer", ""),
            context.get("graph_vector_answer", "")
        )
    except ValueError as e:
        log.error(e)
        raise gr.Error(str(e))
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error(e)
        raise gr.Error(f"An unexpected error occurred: {str(e)}")


def build_kg(file, schema, example_prompt, build_mode):  # pylint: disable=too-many-branches
    full_path = file.name
    if full_path.endswith(".txt"):
        with open(full_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif full_path.endswith(".docx"):
        text = ""
        doc = docx.Document(full_path)
        for para in doc.paragraphs:
            text += para.text
            text += "\n"
    elif full_path.endswith(".pdf"):
        raise gr.Error("PDF will be supported later! Try to upload text/docx now")
    else:
        raise gr.Error("Please input txt or docx file.")
    builder = KgBuilder(LLMs().get_llm(), Embeddings().get_embedding(), get_hg_client())

    if build_mode != "Rebuild vertex index":
        if schema:
            try:
                schema = json.loads(schema.strip())
                builder.import_schema(from_user_defined=schema)
            except json.JSONDecodeError as e:
                log.error(e)
                builder.import_schema(from_hugegraph=schema)
        else:
            return "ERROR: please input schema."
    builder.chunk_split(text, "paragraph", "zh")

    # TODO: avoid hardcoding the "build_mode" strings (use var/constant instead)
    if build_mode == "Rebuild Vector":
        builder.fetch_graph_data()
    else:
        builder.extract_info(example_prompt, "property_graph")
    # "Test Mode", "Import Mode", "Clear and Import", "Rebuild Vector"
    if build_mode != "Test Mode":
        if build_mode in ("Clear and Import", "Rebuild Vector"):
            clean_vector_index()
        builder.build_vector_index()
    if build_mode == "Clear and Import":
        clean_hg_data()
    if build_mode in ("Clear and Import", "Import Mode"):
        builder.commit_to_hugegraph()
    if build_mode != "Test Mode":
        builder.build_vertex_id_semantic_index()
    log.debug(builder.operators)
    try:
        context = builder.run()
        return context
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error(e)
        raise gr.Error(str(e))


class RAGRequest(BaseModel):
    query: str
    raw_llm: Optional[bool] = None
    vector_only: Optional[bool] = None
    graph_only: Optional[bool] = None
    graph_vector: Optional[bool] = None

class GraphConfigRequest(BaseModel):
    ip: str = "127.0.0.1"
    port: str = "8080"
    name: str = "hugegraph"
    user: str = "xxx"
    pwd: str = "xxx"
    gs: str = ""

class LLMConfigRequest(BaseModel):
    llm_type: str
    # The common parameters shared by OpenAI, Qianfan Wenxin, 
    # and OLLAMA platforms.
    api_key: str
    api_base: str
    language_model: str
    # Openai-only properties
    max_tokens: str = None
    # qianfan-wenxin-only properties
    secret_key: str = None
    # ollama-only properties
    host: str = None
    port: str = None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host")
    parser.add_argument("--port", type=int, default=8001, help="port")
    args = parser.parse_args()
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
                gr.Textbox(value=settings.graph_name, label="graph"),
                gr.Textbox(value=settings.graph_user, label="user"),
                gr.Textbox(value=settings.graph_pwd, label="pwd", type="password"),
                gr.Textbox(value=settings.graph_space, label="graphspace (None)"),
            ]
        graph_config_button = gr.Button("apply configuration")


        def test_api_connection(url, method="GET", headers=None, body=None, auth=None):
            # TODO: use fastapi.request / starlette instead? (Also add a try-catch here)
            log.debug("Request URL: %s", url)
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=5, auth=auth)
                
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=body, timeout=5, auth=auth)
            else:
                log.error("Unsupported method: %s", method)

            if response is None:
                # Unsupported method encountered
                return -1

            # HTTP API return status
            status_code = response.status_code

            if 200 <= status_code < 300:
                message = "Connection successful. Configured finished."
                log.info(message)
                gr.Info(message)
            else:
                message = f"Connection failed with status code: {status_code}"
                log.error(message)
                gr.Error(message)
            
            return status_code




        def apply_graph_configuration(ip, port, name, user, pwd, gs):
            settings.graph_ip = ip
            settings.graph_port = int(port)
            settings.graph_name = name
            settings.graph_user = user
            settings.graph_pwd = pwd
            settings.graph_space = gs
            # Test graph connection (Auth)
            if gs and gs.strip():
                test_url = f"http://{ip}:{port}/graphspaces/{gs}/graphs/{name}/schema"
            else:
                test_url = f"http://{ip}:{port}/graphs/{name}/schema"
            auth = HTTPBasicAuth(user, pwd)
            # for http api return status
            result = test_api_connection(test_url, auth=auth)
            settings.update_env()
            return result

        graph_config_button.click(apply_graph_configuration, inputs=graph_config_input)  # pylint: disable=no-member

        gr.Markdown("2. Set up the LLM.")
        llm_dropdown = gr.Dropdown(
            choices=["openai", "qianfan_wenxin", "ollama"],
            value=settings.llm_type,
            label="LLM"
        )

        # Different llm models have different parameters,
        # so no meaningful argument names are given here
        def apply_llm_configuration(arg1, arg2, arg3, arg4):
            llm_option = settings.llm_type
            status_code = 200
            if llm_option == "openai":
                settings.openai_api_key = arg1
                settings.openai_api_base = arg2
                settings.openai_language_model = arg3
                settings.openai_max_tokens = int(arg4)
                test_url = settings.openai_api_base + "/models"
                headers = {"Authorization": f"Bearer {arg1}"}
                status_code = test_api_connection(test_url, headers=headers)
            elif llm_option == "qianfan_wenxin":
                settings.qianfan_api_key = arg1
                settings.qianfan_secret_key = arg2
                settings.qianfan_language_model = arg3
                # TODO: test the connection
                # test_url = "https://aip.baidubce.com/oauth/2.0/token"  # POST
            elif llm_option == "ollama":
                settings.ollama_host = arg1
                settings.ollama_port = int(arg2)
                settings.ollama_language_model = arg3
            gr.Info("configured!")
            settings.update_env()
            return status_code

        @gr.render(inputs=[llm_dropdown])
        def llm_settings(llm_type):
            settings.llm_type = llm_type
            if llm_type == "openai":
                with gr.Row():
                    llm_config_input = [
                        gr.Textbox(value=settings.openai_api_key, label="api_key", type="password"),
                        gr.Textbox(value=settings.openai_api_base, label="api_base"),
                        gr.Textbox(value=settings.openai_language_model, label="model_name"),
                        gr.Textbox(value=settings.openai_max_tokens, label="max_token"),
                    ]
            elif llm_type == "ollama":
                with gr.Row():
                    llm_config_input = [
                        gr.Textbox(value=settings.ollama_host, label="host"),
                        gr.Textbox(value=str(settings.ollama_port), label="port"),
                        gr.Textbox(value=settings.ollama_language_model, label="model_name"),
                        gr.Textbox(value="", visible=False)
                    ]
            elif llm_type == "qianfan_wenxin":
                with gr.Row():
                    llm_config_input = [
                        gr.Textbox(value=settings.qianfan_api_key, label="api_key",
                                type="password"),
                        gr.Textbox(value=settings.qianfan_secret_key, label="secret_key",
                                type="password"),
                        gr.Textbox(value=settings.qianfan_language_model, label="model_name"),
                        gr.Textbox(value="", visible=False)
                    ]
                log.debug(llm_config_input)
            else:
                llm_config_input = []
            llm_config_button = gr.Button("apply configuration")

            llm_config_button.click(apply_llm_configuration, inputs=llm_config_input)  # pylint: disable=no-member

        gr.Markdown("3. Set up the Embedding.")
        embedding_dropdown = gr.Dropdown(
            choices=["openai", "ollama", "qianfan_wenxin"],
            value=settings.embedding_type,
            label="Embedding"
        )




        def apply_embedding_configuration(embedding_option, arg1, arg2, arg3):
            if embedding_option == "openai":
                settings.openai_api_key = arg1
                settings.openai_api_base = arg2
                settings.openai_embedding_model = arg3
                test_url = settings.openai_api_base + "/models"
                headers = {"Authorization": f"Bearer {arg1}"}
                test_api_connection(test_url, headers=headers)
            elif embedding_option == "ollama":
                settings.ollama_host = arg1
                settings.ollama_port = int(arg2)
                settings.ollama_embedding_model = arg3
            elif embedding_option == "qianfan_wenxin":
                settings.qianfan_access_token = arg1
                settings.qianfan_embed_url = arg2
            settings.update_env()
            gr.Info("configured!")

        @gr.render(inputs=[embedding_dropdown])
        def embedding_settings(embedding_type):
            settings.embedding_type = embedding_type
            if embedding_type == "openai":
                with gr.Row():
                    embedding_config_input = [
                        gr.Textbox(value=settings.openai_api_key, label="api_key", type="password"),
                        gr.Textbox(value=settings.openai_api_base, label="api_base"),
                        gr.Textbox(value=settings.openai_embedding_model, label="model_name")
                    ]
            elif embedding_type == "qianfan_wenxin":
                with gr.Row():
                    embedding_config_input = [
                        gr.Textbox(value=settings.qianfan_api_key, label="api_key",
                                type="password"),
                        gr.Textbox(value=settings.qianfan_secret_key, label="secret_key",
                                type="password"),
                        gr.Textbox(value=settings.qianfan_embedding_model, label="model_name"),
                    ]
            elif embedding_type == "ollama":
                with gr.Row():
                    embedding_config_input = [
                        gr.Textbox(value=settings.ollama_host, label="host"),
                        gr.Textbox(value=str(settings.ollama_port), label="port"),
                        gr.Textbox(value=settings.ollama_embedding_model, label="model_name"),
                    ]
            else:
                embedding_config_input = []

            embedding_config_button = gr.Button("apply configuration")
            
            # 在这里调用独立的 apply_embedding_configuration 函数
            embedding_config_button.click(
                lambda arg1, arg2, arg3: apply_embedding_configuration(settings.embedding_type, arg1, arg2, arg3),
                inputs=embedding_config_input
            )


            embedding_config_button.click(apply_embedding_configuration,  # pylint: disable=no-member
                                          inputs=embedding_config_input)


        gr.Markdown(
            """## 1. Build vector/graph RAG (💡)
- Document: Input document file which should be TXT or DOCX.
- Schema: Accepts two types of text as below:
    - User-defined JSON format Schema.
    - Specify the name of the HugeGraph graph instance, it will automatically get the schema from it.
- Info extract head: The head of prompt of info extracting.
- Build mode: 
    - Test Mode: Only extract vertices and edges from the file into memory (without building the vector index or 
    writing data into HugeGraph)
    - Import Mode: Extract the data and append it to HugeGraph & the vector index (without clearing any existing data)
    - Clear and Import: Clear all existed RAG data(vector + graph), then rebuild them from the current input
    - Rebuild Vector: Only rebuild vector index. (keep the graph data intact)
"""
        )

        SCHEMA = """{
  "vertexlabels": [
    {
      "id":1,
      "name": "person",
      "id_strategy": "PRIMARY_KEY",
      "primary_keys":["name"],
      "properties": ["name","age","occupation"]
    },
    {
      "id":2,
      "name": "webpage",
      "id_strategy":"PRIMARY_KEY",
      "primary_keys":["name"],
      "properties": ["name","url"]
    }
  ],
  "edgelabels": [
    {
      "id": 1,
      "name": "roommate",
      "source_label": "person",
      "target_label": "person",
      "properties": ["date"]
    },
    {
      "id": 2,
      "name": "link",
      "source_label": "webpage",
      "target_label": "person",
      "properties": []
    }
  ]
}"""

        with gr.Row():
            input_file = gr.File(value=os.path.join(resource_path, "demo", "test.txt"),
                                 label="Document")
            input_schema = gr.Textbox(value=SCHEMA, label="Schema")
            info_extract_template = gr.Textbox(value=SCHEMA_EXAMPLE_PROMPT,
                                               label="Info extract head")
            with gr.Column():
                mode = gr.Radio(choices=["Test Mode", "Import Mode", "Clear and Import", "Rebuild Vector"],
                                value="Test Mode", label="Build mode")
                btn = gr.Button("Build Vector/Graph RAG")
        with gr.Row():
            out = gr.Textbox(label="Output", show_copy_button=True)
        btn.click(  # pylint: disable=no-member
            fn=build_kg,
            inputs=[input_file, input_schema, info_extract_template, mode],
            outputs=out
        )

        gr.Markdown("""## 2. RAG with HugeGraph 📖""")
        with gr.Row():
            with gr.Column(scale=2):
                inp = gr.Textbox(value="Tell me about Sarah.", label="Question", show_copy_button=True)
                raw_out = gr.Textbox(label="Basic LLM Answer", show_copy_button=True)
                vector_only_out = gr.Textbox(label="Vector-only Answer", show_copy_button=True)
                graph_only_out = gr.Textbox(label="Graph-only Answer", show_copy_button=True)
                graph_vector_out = gr.Textbox(label="Graph-Vector Answer", show_copy_button=True)
            with gr.Column(scale=1):
                raw_radio = gr.Radio(choices=[True, False], value=True,
                                     label="Basic LLM Answer")
                vector_only_radio = gr.Radio(choices=[True, False], value=False,
                                             label="Vector-only Answer")
                graph_only_radio = gr.Radio(choices=[True, False], value=False,
                                            label="Graph-only Answer")
                graph_vector_radio = gr.Radio(choices=[True, False], value=False,
                                              label="Graph-Vector Answer")
                btn = gr.Button("Answer Question")
        btn.click(fn=graph_rag,
                  inputs=[inp, raw_radio, vector_only_radio, graph_only_radio,  # pylint: disable=no-member
                          graph_vector_radio],
                  outputs=[raw_out, vector_only_out, graph_only_out, graph_vector_out])

        gr.Markdown("""## 3. Others (🚧) """)
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(value="g.V().limit(10)", label="Gremlin query", show_copy_button=True)
                format = gr.Checkbox(label="Format JSON", value=True)
            out = gr.Textbox(label="Output", show_copy_button=True)
        btn = gr.Button("Run gremlin query on HugeGraph")
        btn.click(fn=run_gremlin_query, inputs=[inp, format], outputs=out)  # pylint: disable=no-member

        with gr.Row():
            inp = []
            out = gr.Textbox(label="Output", show_copy_button=True)
        btn = gr.Button("(BETA) Init HugeGraph test data (🚧WIP)")
        btn.click(fn=init_hg_test_data, inputs=inp, outputs=out)  # pylint: disable=no-member

    @app.post("/rag")
    def graph_rag_api(req: RAGRequest):
        result = graph_rag(req.query, req.raw_llm, req.vector_only, req.graph_only, req.graph_vector)
        return {key: value for key, value in zip(
            ["raw_llm", "vector_only", "graph_only", "graph_vector"], result) if getattr(req, key)}

    @app.post("/graph/config")
    def graph_config_api(req: GraphConfigRequest):
        # Accept status code
        status_code = apply_graph_configuration(req.ip, req.port, req.name, req.user, req.pwd, req.gs)
        
        if status_code == -1:
            return {"message":"Unsupported HTTP method"}
        
        if 200 <= status_code < 300:
            return {"message":"Connection successful. Configured finished."}
        else:
            return {"message":f"Connection failed with status code: {status_code}"}
    
    @app.post("/llm/config")
    def graph_config_api(req: LLMConfigRequest):
        settings.llm_type = req.llm_type
        
        if req.llm_type == "openai":
            status_code = apply_llm_configuration(req.api_key, req.api_base, req.language_model, req.max_tokens)
        elif req.llm_type == "qianfan_wenxin":
            status_code = apply_llm_configuration(req.api_key, req.secret_key, req.language_model, None)
        else:
            status_code = apply_llm_configuration(req.host, req.port, req.language_model, None)
        
        if status_code == -1:
            return {"message":"Unsupported HTTP method"}

        if 200 <= status_code < 300:
            return {"message":"Connection successful. Configured finished."}
        else:
            return {"message":f"Connection failed with status code: {status_code}"}


    app = gr.mount_gradio_app(app, hugegraph_llm, path="/")
    # Note: set reload to False in production environment
    uvicorn.run(app, host=args.host, port=args.port)
    # TODO: we can't use reload now due to the config 'app' of uvicorn.run
    # uvicorn.run("rag_web_demo:app", host="0.0.0.0", port=8001, reload=True)
