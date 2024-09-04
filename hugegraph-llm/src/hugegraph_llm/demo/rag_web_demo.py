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


import argparse
import json
import os
from typing import List

import docx
import gradio as gr
import requests
import uvicorn
from fastapi import FastAPI, Depends, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from gradio.utils import NamedString
from requests.auth import HTTPBasicAuth

from hugegraph_llm.demo.css import CSS
from hugegraph_llm.api.rag_api import rag_http_api
from hugegraph_llm.config import settings, resource_path
from hugegraph_llm.enums.build_mode import BuildMode
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.graph_rag_task import RAGPipeline
from hugegraph_llm.operators.kg_construction_task import KgBuilder
from hugegraph_llm.operators.llm_op.property_graph_extract import SCHEMA_EXAMPLE_PROMPT
from hugegraph_llm.utils.graph_index_utils import get_graph_index_info, clean_graph_index, fit_vid_index, \
    build_graph_index, extract_graph, import_graph_data
from hugegraph_llm.utils.hugegraph_utils import get_hg_client
from hugegraph_llm.utils.hugegraph_utils import init_hg_test_data, run_gremlin_query, clean_hg_data
from hugegraph_llm.utils.log import log
from hugegraph_llm.utils.vector_index_utils import clean_vector_index, build_vector_index, get_vector_index_info

sec = HTTPBearer()


def authenticate(credentials: HTTPAuthorizationCredentials = Depends(sec)):
    correct_token = os.getenv("TOKEN")
    if credentials.credentials != correct_token:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token {credentials.credentials}, please contact the admin",
            headers={"WWW-Authenticate": "Bearer"},
        )


def rag_answer(
        text: str, raw_answer: bool, vector_only_answer: bool, graph_only_answer: bool,
         graph_vector_answer: bool, answer_prompt: str) -> tuple:
    vector_search = vector_only_answer or graph_vector_answer
    graph_search = graph_only_answer or graph_vector_answer

    if raw_answer is False and not vector_search and not graph_search:
        gr.Warning("Please select at least one generate mode.")
        return "", "", "", ""
    searcher = RAGPipeline()
    if vector_search:
        searcher.query_vector_index_for_rag()
    if graph_search:
        searcher.extract_keyword().match_keyword_to_id().query_graph_for_rag()
    # TODO: add more user-defined search strategies
    searcher.merge_dedup_rerank().synthesize_answer(
        raw_answer=raw_answer,
        vector_only_answer=vector_only_answer,
        graph_only_answer=graph_only_answer,
        graph_vector_answer=graph_vector_answer,
        answer_prompt=answer_prompt
    )

    try:
        context = searcher.run(verbose=True, query=text)
        return (
            context.get("raw_answer", ""),
            context.get("vector_only_answer", ""),
            context.get("graph_only_answer", ""),
            context.get("graph_vector_answer", ""),
        )
    except ValueError as e:
        log.error(e)
        raise gr.Error(str(e))
    except Exception as e:
        log.error(e)
        raise gr.Error(f"An unexpected error occurred: {str(e)}")


def build_kg(
        input_file: List[NamedString],
        input_text: str,
        schema: str,
        example_prompt: str,
        build_mode: str
) -> str:
    if input_file:
        texts = []
        for file in input_file:
            full_path = file.name
            if full_path.endswith(".txt"):
                with open(full_path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            elif full_path.endswith(".docx"):
                text = ""
                doc = docx.Document(full_path)
                for para in doc.paragraphs:
                    text += para.text
                    text += "\n"
                texts.append(text)
            elif full_path.endswith(".pdf"):
                # TODO: support PDF file
                raise gr.Error("PDF will be supported later! Try to upload text/docx now")
            else:
                raise gr.Error("Please input txt or docx file.")
    elif input_text:
        texts = [input_text]
    else:
        raise gr.Error("Please input text or upload file.")
    if build_mode in (BuildMode.CLEAR_AND_IMPORT.value, BuildMode.REBUILD_VECTOR.value):
        clean_vector_index()
    if build_mode == BuildMode.CLEAR_AND_IMPORT.value:
        clean_hg_data()
    builder = KgBuilder(LLMs().get_llm(), Embeddings().get_embedding(), get_hg_client())

    if build_mode != BuildMode.REBUILD_VERTEX_INDEX.value:
        if schema:
            try:
                schema = json.loads(schema.strip())
                builder.import_schema(from_user_defined=schema)
            except json.JSONDecodeError as e:
                log.error(e)
                builder.import_schema(from_hugegraph=schema)
        else:
            return "ERROR: please input schema."
    builder.chunk_split(texts, "paragraph", "zh")

    if build_mode == BuildMode.REBUILD_VECTOR.value:
        builder.fetch_graph_data()
    else:
        builder.extract_info(example_prompt, "property_graph")

    # "Test Mode", "Import Mode", "Clear and Import", "Rebuild Vector"
    if build_mode != BuildMode.TEST_MODE.value:
        builder.build_vector_index()
    if build_mode in (BuildMode.CLEAR_AND_IMPORT.value, BuildMode.IMPORT_MODE.value):
        builder.commit_to_hugegraph()
    if build_mode != BuildMode.TEST_MODE.value:
        builder.build_vertex_id_semantic_index()
    log.warning("Current building mode: [%s]", build_mode)
    try:
        context = builder.run()
        return str(context)
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error(e)
        raise gr.Error(str(e))


def test_api_connection(url, method="GET",
                        headers=None, params=None, body=None, auth=None, origin_call=None) -> int:
    # TODO: use fastapi.request / starlette instead?
    log.debug("Request URL: %s", url)
    try:
        if method.upper() == "GET":
            resp = requests.get(url, headers=headers, params=params, timeout=5, auth=auth)
        elif method.upper() == "POST":
            resp = requests.post(url, headers=headers, params=params, json=body, timeout=5, auth=auth)
        else:
            raise ValueError("Unsupported HTTP method, please use GET/POST instead")
    except requests.exceptions.RequestException as e:
        msg = f"Connection failed: {e}"
        log.error(msg)
        if origin_call is None:
            raise gr.Error(msg)
        return -1  # Error code

    if 200 <= resp.status_code < 300:
        msg = "Test connection successful~"
        log.info(msg)
        gr.Info(msg)
    else:
        msg = f"Connection failed with status code: {resp.status_code}, error: {resp.text}"
        log.error(msg)
        # TODO: Only the message returned by rag can be processed, and the other return values can't be processed
        if origin_call is None:
            raise gr.Error(json.loads(resp.text).get("message", msg))
    return resp.status_code


def config_qianfan_model(arg1, arg2, arg3=None, origin_call=None) -> int:
    settings.qianfan_api_key = arg1
    settings.qianfan_secret_key = arg2
    settings.qianfan_language_model = arg3
    params = {
        "grant_type": "client_credentials",
        "client_id": arg1,
        "client_secret": arg2
    }
    status_code = test_api_connection("https://aip.baidubce.com/oauth/2.0/token", "POST", params=params,
                                      origin_call=origin_call)
    return status_code


def apply_embedding_config(arg1, arg2, arg3, origin_call=None) -> int:
    status_code = -1
    embedding_option = settings.embedding_type
    if embedding_option == "openai":
        settings.openai_api_key = arg1
        settings.openai_api_base = arg2
        settings.openai_embedding_model = arg3
        test_url = settings.openai_api_base + "/models"
        headers = {"Authorization": f"Bearer {arg1}"}
        status_code = test_api_connection(test_url, headers=headers, origin_call=origin_call)
    elif embedding_option == "qianfan_wenxin":
        status_code = config_qianfan_model(arg1, arg2, origin_call=origin_call)
        settings.qianfan_embedding_model = arg3
    elif embedding_option == "ollama":
        settings.ollama_host = arg1
        settings.ollama_port = int(arg2)
        settings.ollama_embedding_model = arg3
        status_code = test_api_connection(f"http://{arg1}:{arg2}", origin_call=origin_call)
    settings.update_env()
    gr.Info("Configured!")
    return status_code


def apply_graph_config(ip, port, name, user, pwd, gs, origin_call=None) -> int:
    settings.graph_ip = ip
    settings.graph_port = port
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
    response = test_api_connection(test_url, auth=auth, origin_call=origin_call)
    settings.update_env()
    return response


# Different llm models have different parameters,
# so no meaningful argument names are given here
def apply_llm_config(arg1, arg2, arg3, arg4, origin_call=None) -> int:
    llm_option = settings.llm_type
    status_code = -1
    if llm_option == "openai":
        settings.openai_api_key = arg1
        settings.openai_api_base = arg2
        settings.openai_language_model = arg3
        settings.openai_max_tokens = int(arg4)
        test_url = settings.openai_api_base + "/models"
        headers = {"Authorization": f"Bearer {arg1}"}
        status_code = test_api_connection(test_url, headers=headers, origin_call=origin_call)
    elif llm_option == "qianfan_wenxin":
        status_code = config_qianfan_model(arg1, arg2, arg3, origin_call)
    elif llm_option == "ollama":
        settings.ollama_host = arg1
        settings.ollama_port = int(arg2)
        settings.ollama_language_model = arg3
        status_code = test_api_connection(f"http://{arg1}:{arg2}", origin_call=origin_call)
    gr.Info("Configured!")
    settings.update_env()
    return status_code


def init_rag_ui() -> gr.Interface:
    with gr.Blocks(theme='default',
                   title="HugeGraph RAG Platform",
                   css=CSS) as hugegraph_llm_ui:
        gr.Markdown(
            """# HugeGraph LLM RAG Demo
        1. Set up the HugeGraph server."""
        )
        with gr.Row():
            graph_config_input = [
                gr.Textbox(value=settings.graph_ip, label="ip"),
                gr.Textbox(value=settings.graph_port, label="port"),
                gr.Textbox(value=settings.graph_name, label="graph"),
                gr.Textbox(value=settings.graph_user, label="user"),
                gr.Textbox(value=settings.graph_pwd, label="pwd", type="password"),
                gr.Textbox(value=settings.graph_space, label="graphspace(Optional)"),
            ]
        graph_config_button = gr.Button("apply configuration")

        graph_config_button.click(apply_graph_config, inputs=graph_config_input)  # pylint: disable=no-member

        gr.Markdown("2. Set up the LLM.")
        llm_dropdown = gr.Dropdown(choices=["openai", "qianfan_wenxin", "ollama"], value=settings.llm_type, label="LLM")

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
                        gr.Textbox(value="", visible=False),
                    ]
            elif llm_type == "qianfan_wenxin":
                with gr.Row():
                    llm_config_input = [
                        gr.Textbox(value=settings.qianfan_api_key, label="api_key", type="password"),
                        gr.Textbox(value=settings.qianfan_secret_key, label="secret_key", type="password"),
                        gr.Textbox(value=settings.qianfan_language_model, label="model_name"),
                        gr.Textbox(value="", visible=False),
                    ]
                log.debug(llm_config_input)
            else:
                llm_config_input = []
            llm_config_button = gr.Button("apply configuration")

            llm_config_button.click(apply_llm_config, inputs=llm_config_input)  # pylint: disable=no-member

        gr.Markdown("3. Set up the Embedding.")
        embedding_dropdown = gr.Dropdown(
            choices=["openai", "qianfan_wenxin", "ollama"], value=settings.embedding_type, label="Embedding"
        )

        @gr.render(inputs=[embedding_dropdown])
        def embedding_settings(embedding_type):
            settings.embedding_type = embedding_type
            if embedding_type == "openai":
                with gr.Row():
                    embedding_config_input = [
                        gr.Textbox(value=settings.openai_api_key, label="api_key", type="password"),
                        gr.Textbox(value=settings.openai_api_base, label="api_base"),
                        gr.Textbox(value=settings.openai_embedding_model, label="model_name"),
                    ]
            elif embedding_type == "qianfan_wenxin":
                with gr.Row():
                    embedding_config_input = [
                        gr.Textbox(value=settings.qianfan_api_key, label="api_key", type="password"),
                        gr.Textbox(value=settings.qianfan_secret_key, label="secret_key", type="password"),
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

            # Call the separate apply_embedding_configuration function here
            embedding_config_button.click(  # pylint: disable=no-member
                apply_embedding_config, inputs=embedding_config_input  # pylint: disable=no-member
            )

        gr.Markdown(
            """## 1. Build vector/graph RAG (💡)
- Doc(s): Upload document file(s) which should be TXT or DOCX. (Multiple files can be selected together)
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

        schema = """{
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
            with gr.Column():
                with gr.Tab("file") as tab_upload_file:
                    input_file = gr.File(
                        value=[os.path.join(resource_path, "demo", "test.txt")],
                        label="Docs (multi-files can be selected together)",
                        file_count="multiple"
                    )
                with gr.Tab("text") as tab_upload_text:
                    input_text = gr.Textbox(value="", label="Doc(s)")
            input_schema = gr.Textbox(value=schema, label="Schema")
            info_extract_template = gr.Textbox(value=SCHEMA_EXAMPLE_PROMPT, label="Info extract head")
        with gr.Column(visible=False, elem_classes="modal-box") as preview_box:
            with gr.Row():
                import_btn = gr.Button("Import Into Graph")
                close_btn = gr.Button("Cancel")

            with gr.Row():
                extraction_result = gr.Textbox(label="Extraction Result")

        with gr.Row():
            vector_index_btn0 = gr.Button("Get Vector Index Info")
            vector_index_btn1 = gr.Button("Clear Vector Index")
            vector_index_btn2 = gr.Button("Import into Vector Index", variant="primary")
        with gr.Row():
            graph_index_btn0 = gr.Button("Get Graph Index Info")
            graph_index_btn1 = gr.Button("Clear Graph Index")
            graph_index_btn2 = gr.Button("Extract Graph", variant="primary")
            graph_index_btn3 = gr.Button("Fit Vid Index")
        with gr.Row():
            out = gr.Textbox(label="Output", show_copy_button=True)
        vector_index_btn0.click(get_vector_index_info, outputs=out)  # pylint: disable=no-member
        vector_index_btn1.click(clean_vector_index)  # pylint: disable=no-member
        vector_index_btn2.click(build_vector_index, inputs=[input_file, input_text], outputs=out)  # pylint: disable=no-member
        graph_index_btn0.click(get_graph_index_info, outputs=out)  # pylint: disable=no-member
        graph_index_btn1.click(clean_graph_index)  # pylint: disable=no-member
        graph_index_btn2.click(extract_graph, inputs=[input_file, input_text, input_schema,  # pylint: disable=no-member
                                                      info_extract_template], outputs=[out, extraction_result, preview_box])
        graph_index_btn3.click(fit_vid_index, outputs=out)  # pylint: disable=no-member

        import_btn.click(import_graph_data, inputs=[extraction_result], outputs=[preview_box, out], queue=False)  # pylint: disable=no-member
        close_btn.click(lambda: gr.Column(visible=False), outputs=[preview_box], queue=False)  # pylint: disable=no-member


        def on_tab_select(input_f, input_t, evt: gr.SelectData):
            print(f"You selected {evt.value} at {evt.index} from {evt.target}")
            if evt.value == "file":
                return input_f, ""
            else:
                return [], input_t
        tab_upload_file.select(fn=on_tab_select, inputs=[input_file, input_text], outputs=[input_file, input_text])  # pylint: disable=no-member
        tab_upload_text.select(fn=on_tab_select, inputs=[input_file, input_text], outputs=[input_file, input_text])  # pylint: disable=no-member


        gr.Markdown("""## 2. RAG with HugeGraph 📖""")
        with gr.Row():
            with gr.Column(scale=2):
                inp = gr.Textbox(value="Tell me about Sarah.", label="Question", show_copy_button=True)
                raw_out = gr.Textbox(label="Basic LLM Answer", show_copy_button=True)
                vector_only_out = gr.Textbox(label="Vector-only Answer", show_copy_button=True)
                graph_only_out = gr.Textbox(label="Graph-only Answer", show_copy_button=True)
                graph_vector_out = gr.Textbox(label="Graph-Vector Answer", show_copy_button=True)
            with gr.Column(scale=1):
                raw_radio = gr.Radio(choices=[True, False], value=True, label="Basic LLM Answer")
                vector_only_radio = gr.Radio(choices=[True, False], value=False, label="Vector-only Answer")
                graph_only_radio = gr.Radio(choices=[True, False], value=False, label="Graph-only Answer")
                graph_vector_radio = gr.Radio(choices=[True, False], value=False, label="Graph-Vector Answer")
                btn = gr.Button("Answer Question")
                from hugegraph_llm.operators.llm_op.answer_synthesize import DEFAULT_ANSWER_TEMPLATE
                answer_prompt_input = gr.Textbox(value=DEFAULT_ANSWER_TEMPLATE, label="Custom Prompt",
                                                 show_copy_button=True)
        btn.click(  # pylint: disable=no-member
            fn=rag_answer,
            inputs=[
                inp,
                raw_radio,
                vector_only_radio,
                graph_only_radio,
                graph_vector_radio,
                answer_prompt_input,
            ],
            outputs=[raw_out, vector_only_out, graph_only_out, graph_vector_out],
        )

        gr.Markdown("""## 3. Others (🚧) """)
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(value="g.V().limit(10)", label="Gremlin query", show_copy_button=True)
                fmt = gr.Checkbox(label="Format JSON", value=True)
            out = gr.Textbox(label="Output", show_copy_button=True)
        btn = gr.Button("Run gremlin query on HugeGraph")
        btn.click(fn=run_gremlin_query, inputs=[inp, fmt], outputs=out)  # pylint: disable=no-member

        with gr.Row():
            inp = []
            out = gr.Textbox(label="Output", show_copy_button=True)
        btn = gr.Button("(BETA) Init HugeGraph test data (🚧WIP)")
        btn.click(fn=init_hg_test_data, inputs=inp, outputs=out)  # pylint: disable=no-member
    return hugegraph_llm_ui


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host")
    parser.add_argument("--port", type=int, default=8001, help="port")
    args = parser.parse_args()
    app = FastAPI()
    app_auth = APIRouter(dependencies=[Depends(authenticate)])

    hugegraph_llm = init_rag_ui()
    rag_http_api(app_auth, rag_answer, apply_graph_config, apply_llm_config, apply_embedding_config)

    app.include_router(app_auth)
    auth_enabled = os.getenv("ENABLE_LOGIN", "False").lower() == "true"
    log.info("Authentication is %s.", "enabled" if auth_enabled else "disabled")
    # TODO: support multi-user login when need

    app = gr.mount_gradio_app(app, hugegraph_llm, path="/", auth=("rag", os.getenv("TOKEN")) if auth_enabled else None)

    # TODO: we can't use reload now due to the config 'app' of uvicorn.run
    # ❎:f'{__name__}:app' / rag_web_demo:app / hugegraph_llm.demo.rag_web_demo:app
    # TODO: merge unicorn log to avoid duplicate log output (should be unified/fixed later)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
