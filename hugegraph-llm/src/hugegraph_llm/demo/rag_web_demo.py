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
from typing import Tuple, Literal, Optional

import gradio as gr
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI, Depends, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from gradio.utils import NamedString
from requests.auth import HTTPBasicAuth

from hugegraph_llm.api.rag_api import rag_http_api
from hugegraph_llm.config import settings, resource_path, prompt
from hugegraph_llm.operators.graph_rag_task import RAGPipeline
from hugegraph_llm.operators.llm_op.property_graph_extract import SCHEMA_EXAMPLE_PROMPT
from hugegraph_llm.resources.demo.css import CSS
from hugegraph_llm.utils.graph_index_utils import get_graph_index_info, clean_all_graph_index, fit_vid_index, \
    extract_graph, import_graph_data
from hugegraph_llm.utils.hugegraph_utils import init_hg_test_data, run_gremlin_query
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
    text: str,
    raw_answer: bool,
    vector_only_answer: bool,
    graph_only_answer: bool,
    graph_vector_answer: bool,
    graph_ratio: float,
    rerank_method: Literal["bleu", "reranker"],
    near_neighbor_first: bool,
    custom_related_information: str,
    answer_prompt: str,
) -> Tuple:
    """
    Generate an answer using the RAG (Retrieval-Augmented Generation) pipeline.
    1. Initialize the RAGPipeline.
    2. Select vector search or graph search based on parameters.
    3. Merge, deduplicate, and rerank the results.
    4. Synthesize the final answer.
    5. Run the pipeline and return the results.
    """
    should_update_prompt = prompt.default_question != text or prompt.answer_prompt != answer_prompt
    if should_update_prompt or prompt.custom_rerank_info != custom_related_information:
        prompt.custom_rerank_info = custom_related_information
        prompt.default_question = text
        prompt.answer_prompt = answer_prompt
        prompt.update_yaml_file()
    
    vector_search = vector_only_answer or graph_vector_answer
    graph_search = graph_only_answer or graph_vector_answer
    if raw_answer is False and not vector_search and not graph_search:
        gr.Warning("Please select at least one generate mode.")
        return "", "", "", ""

    rag = RAGPipeline()
    if vector_search:
        rag.query_vector_index()
    if graph_search:
        rag.extract_keywords().keywords_to_vid().query_graphdb()
    # TODO: add more user-defined search strategies
    rag.merge_dedup_rerank(graph_ratio, rerank_method, near_neighbor_first, custom_related_information)
    rag.synthesize_answer(raw_answer, vector_only_answer, graph_only_answer, graph_vector_answer, answer_prompt)

    try:
        context = rag.run(verbose=True, query=text, vector_search=vector_search, graph_search=graph_search)
        if context.get("switch_to_bleu"):
            gr.Warning("Online reranker fails, automatically switches to local bleu rerank.")
        return (
            context.get("raw_answer", ""),
            context.get("vector_only_answer", ""),
            context.get("graph_only_answer", ""),
            context.get("graph_vector_answer", ""),
        )
    except ValueError as e:
        log.critical(e)
        raise gr.Error(str(e))
    except Exception as e:
        log.critical(e)
        raise gr.Error(f"An unexpected error occurred: {str(e)}")


def test_api_connection(url, method="GET", headers=None, params=None, body=None, auth=None, origin_call=None) -> int:
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
            try:
                raise gr.Error(json.loads(resp.text).get("message", msg))
            except json.decoder.JSONDecodeError and AttributeError:
                raise gr.Error(resp.text)
    return resp.status_code


def config_qianfan_model(arg1, arg2, arg3=None, origin_call=None) -> int:
    settings.qianfan_api_key = arg1
    settings.qianfan_secret_key = arg2
    settings.qianfan_language_model = arg3
    params = {
        "grant_type": "client_credentials",
        "client_id": arg1,
        "client_secret": arg2,
    }
    status_code = test_api_connection(
        "https://aip.baidubce.com/oauth/2.0/token", "POST", params=params, origin_call=origin_call
    )
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


def apply_reranker_config(
    reranker_api_key: Optional[str] = None,
    reranker_model: Optional[str] = None,
    cohere_base_url: Optional[str] = None,
    origin_call=None,
) -> int:
    status_code = -1
    reranker_option = settings.reranker_type
    if reranker_option == "cohere":
        settings.reranker_api_key = reranker_api_key
        settings.reranker_model = reranker_model
        settings.cohere_base_url = cohere_base_url
        headers = {"Authorization": f"Bearer {reranker_api_key}"}
        status_code = test_api_connection(
            cohere_base_url.rsplit("/", 1)[0] + "/check-api-key",
            method="POST",
            headers=headers,
            origin_call=origin_call,
        )
    elif reranker_option == "siliconflow":
        settings.reranker_api_key = reranker_api_key
        settings.reranker_model = reranker_model
        from pyhugegraph.utils.constants import Constants
        headers = {
            "accept": Constants.HEADER_CONTENT_TYPE,
            "authorization": f"Bearer {reranker_api_key}",
        }
        status_code = test_api_connection(
            "https://api.siliconflow.cn/v1/user/info",
            headers=headers,
            origin_call=origin_call,
        )
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
    with gr.Blocks(
        theme="default",
        title="HugeGraph RAG Platform",
        css=CSS,
    ) as hugegraph_llm_ui:
        gr.Markdown("# HugeGraph LLM RAG Demo")
        with gr.Accordion("1. Set up the HugeGraph server.", open=False):
            with gr.Row():
                graph_config_input = [
                    gr.Textbox(value=settings.graph_ip, label="ip"),
                    gr.Textbox(value=settings.graph_port, label="port"),
                    gr.Textbox(value=settings.graph_name, label="graph"),
                    gr.Textbox(value=settings.graph_user, label="user"),
                    gr.Textbox(value=settings.graph_pwd, label="pwd", type="password"),
                    gr.Textbox(value=settings.graph_space, label="graphspace(Optional)"),
                ]
            graph_config_button = gr.Button("Apply config")
        graph_config_button.click(apply_graph_config, inputs=graph_config_input)  # pylint: disable=no-member

        with gr.Accordion("2. Set up the LLM.", open=False):
            llm_dropdown = gr.Dropdown(choices=["openai", "qianfan_wenxin", "ollama"],
                                       value=settings.llm_type, label="LLM")

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
                else:
                    llm_config_input = []
                llm_config_button = gr.Button("apply configuration")
                llm_config_button.click(apply_llm_config, inputs=llm_config_input)  # pylint: disable=no-member

        with gr.Accordion("3. Set up the Embedding.", open=False):
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
                    fn=apply_embedding_config,
                    inputs=embedding_config_input,  # pylint: disable=no-member
                )

        with gr.Accordion("4. Set up the Reranker.", open=False):
            reranker_dropdown = gr.Dropdown(
                choices=["cohere", "siliconflow", ("default/offline", "None")],
                value=os.getenv("reranker_type") or "None",
                label="Reranker",
            )

            @gr.render(inputs=[reranker_dropdown])
            def reranker_settings(reranker_type):
                settings.reranker_type = reranker_type if reranker_type != "None" else None
                if reranker_type == "cohere":
                    with gr.Row():
                        reranker_config_input = [
                            gr.Textbox(value=settings.reranker_api_key, label="api_key", type="password"),
                            gr.Textbox(value=settings.reranker_model, label="model"),
                            gr.Textbox(value=settings.cohere_base_url, label="base_url"),
                        ]
                elif reranker_type == "siliconflow":
                    with gr.Row():
                        reranker_config_input = [
                            gr.Textbox(value=settings.reranker_api_key, label="api_key", type="password"),
                            gr.Textbox(
                                value="BAAI/bge-reranker-v2-m3",
                                label="model",
                                info="Please refer to https://siliconflow.cn/pricing",
                            ),
                        ]
                else:
                    reranker_config_input = []
                reranker_config_button = gr.Button("apply configuration")

                # TODO: use "gr.update()" or other way to update the config in time (refactor the click event)
                # Call the separate apply_reranker_configuration function here
                reranker_config_button.click(  # pylint: disable=no-member
                    fn=apply_reranker_config,
                    inputs=reranker_config_input,  # pylint: disable=no-member
                )

        gr.Markdown(
            """## 1. Build vector/graph RAG (💡)
- Doc(s): 
    - text: Build index from plain text.
    - file: Upload document file(s) which should be TXT or DOCX. (Multiple files can be selected together)
- Schema: Accepts two types of text as below:
    - User-defined JSON format Schema.
    - Specify the name of the HugeGraph graph instance, it will automatically get the schema from it.
- Info extract head: The head of prompt of info extracting.
"""
        )

        schema = prompt.graph_schema
        
        with gr.Row():
            with gr.Column():
                with gr.Tab("text") as tab_upload_text:
                    input_text = gr.Textbox(value="", label="Doc(s)", lines=20, show_copy_button=True)
                with gr.Tab("file") as tab_upload_file:
                    input_file = gr.File(
                        value=[os.path.join(resource_path, "demo", "test.txt")],
                        label="Docs (multi-files can be selected together)",
                        file_count="multiple",
                    )
            input_schema = gr.Textbox(value=schema, label="Schema", lines=15, show_copy_button=True)
            info_extract_template = gr.Textbox(value=SCHEMA_EXAMPLE_PROMPT, label="Info extract head", lines=15,
                                               show_copy_button=True)
            out = gr.Code(label="Output", language="json",  elem_classes="code-container-edit")

        with gr.Row():
            with gr.Accordion("Get RAG Info", open=False):
                with gr.Column():
                    vector_index_btn0 = gr.Button("Get Vector Index Info", size="sm")
                    graph_index_btn0 = gr.Button("Get Graph Index Info", size="sm")
            with gr.Accordion("Clear RAG Info", open=False):
                with gr.Column():
                    vector_index_btn1 = gr.Button("Clear Vector Index", size="sm")
                    graph_index_btn1 = gr.Button("Clear Graph Data & Index", size="sm")

            vector_import_bt = gr.Button("Import into Vector", variant="primary")
            graph_index_rebuild_bt = gr.Button("Rebuild vid Index")
            graph_extract_bt = gr.Button("Extract Graph Data (1)", variant="primary")
            graph_loading_bt = gr.Button("Load into GraphDB (2)", interactive=True)

        vector_index_btn0.click(get_vector_index_info, outputs=out)  # pylint: disable=no-member
        vector_index_btn1.click(clean_vector_index)  # pylint: disable=no-member
        vector_import_bt.click(build_vector_index, inputs=[input_file, input_text], outputs=out)  # pylint: disable=no-member
        graph_index_btn0.click(get_graph_index_info, outputs=out)  # pylint: disable=no-member
        graph_index_btn1.click(clean_all_graph_index)  # pylint: disable=no-member
        graph_index_rebuild_bt.click(fit_vid_index, outputs=out)  # pylint: disable=no-member

        # origin_out = gr.Textbox(visible=False)
        graph_extract_bt.click(  # pylint: disable=no-member
            extract_graph,
            inputs=[input_file, input_text, input_schema, info_extract_template],
            outputs=[out]
        )

        graph_loading_bt.click(import_graph_data, inputs=[out, input_schema], outputs=[out])  # pylint: disable=no-member


        def on_tab_select(input_f, input_t, evt: gr.SelectData):
            print(f"You selected {evt.value} at {evt.index} from {evt.target}")
            if evt.value == "file":
                return input_f, ""
            if evt.value == "text":
                return [], input_t
            return [], ""
        tab_upload_file.select(fn=on_tab_select, inputs=[input_file, input_text], outputs=[input_file, input_text])  # pylint: disable=no-member
        tab_upload_text.select(fn=on_tab_select, inputs=[input_file, input_text], outputs=[input_file, input_text])  # pylint: disable=no-member


        gr.Markdown("""## 2. RAG with HugeGraph 📖""")
        with gr.Row():
            with gr.Column(scale=2):
                inp = gr.Textbox(value=prompt.default_question, label="Question", show_copy_button=True, lines=2)
                raw_out = gr.Textbox(label="Basic LLM Answer", show_copy_button=True)
                vector_only_out = gr.Textbox(label="Vector-only Answer", show_copy_button=True)
                graph_only_out = gr.Textbox(label="Graph-only Answer", show_copy_button=True)
                graph_vector_out = gr.Textbox(label="Graph-Vector Answer", show_copy_button=True)
                from hugegraph_llm.operators.llm_op.answer_synthesize import DEFAULT_ANSWER_TEMPLATE
                answer_prompt_input = gr.Textbox(
                    value=DEFAULT_ANSWER_TEMPLATE, label="Custom Prompt", show_copy_button=True, lines=2
                )
            with gr.Column(scale=1):
                with gr.Row():
                    raw_radio = gr.Radio(choices=[True, False], value=True, label="Basic LLM Answer")
                    vector_only_radio = gr.Radio(choices=[True, False], value=False, label="Vector-only Answer")
                with gr.Row():
                    graph_only_radio = gr.Radio(choices=[True, False], value=False, label="Graph-only Answer")
                    graph_vector_radio = gr.Radio(choices=[True, False], value=False, label="Graph-Vector Answer")

                def toggle_slider(enable):
                    return gr.update(interactive=enable)

                with gr.Column():
                    with gr.Row():
                        online_rerank = os.getenv("reranker_type")
                        rerank_method = gr.Dropdown(
                            choices=["bleu", ("rerank (online)", "reranker")] if online_rerank else ["bleu"],
                            value="reranker" if online_rerank else "bleu",
                            label="Rerank method",
                        )
                        graph_ratio = gr.Slider(0, 1, 0.5, label="Graph Ratio", step=0.1, interactive=False)

                    graph_vector_radio.change(toggle_slider, inputs=graph_vector_radio, outputs=graph_ratio)  # pylint: disable=no-member
                    near_neighbor_first = gr.Checkbox(
                        value=False,
                        label="Near neighbor first(Optional)",
                        info="One-depth neighbors > two-depth neighbors",
                    )
                    custom_related_information = gr.Text(
                        prompt.custom_rerank_info,
                        label="Custom related information(Optional)",
                    )
                    btn = gr.Button("Answer Question", variant="primary")

        btn.click(  # pylint: disable=no-member
            fn=rag_answer,
            inputs=[
                inp,
                raw_radio,
                vector_only_radio,
                graph_only_radio,
                graph_vector_radio,
                graph_ratio,
                rerank_method,
                near_neighbor_first,
                custom_related_information,
                answer_prompt_input,
            ],
            outputs=[raw_out, vector_only_out, graph_only_out, graph_vector_out],
        )

        gr.Markdown("""## 3. User Functions """)
        tests_df_headers = [
            "Question",
            "Expected Answer",
            "Basic LLM Answer",
            "Vector-only Answer",
            "Graph-only Answer",
            "Graph-Vector Answer",
        ]
        answers_path = os.path.join(resource_path, "demo", "questions_answers.xlsx")
        questions_path = os.path.join(resource_path, "demo", "questions.xlsx")
        questions_template_path = os.path.join(resource_path, "demo", "questions_template.xlsx")

        def read_file_to_excel(file: NamedString, line_count: Optional[int] = None):
            df = None
            if not file:
                return pd.DataFrame(), 1
            if file.name.endswith(".xlsx"):
                df = pd.read_excel(file.name, nrows=line_count) if file else pd.DataFrame()
            elif file.name.endswith(".csv"):
                df = pd.read_csv(file.name, nrows=line_count) if file else pd.DataFrame()
            df.to_excel(questions_path, index=False)
            if df.empty:
                df = pd.DataFrame([[""] * len(tests_df_headers)], columns=tests_df_headers)
            else:
                df.columns = tests_df_headers
            # truncate the dataframe if it's too long
            if len(df) > 40:
                return df.head(40), 40
            return df, len(df)

        def change_showing_excel(line_count):
            if os.path.exists(answers_path):
                df = pd.read_excel(answers_path, nrows=line_count)
            elif os.path.exists(questions_path):
                df = pd.read_excel(questions_path, nrows=line_count)
            else:
                df = pd.read_excel(questions_template_path, nrows=line_count)
            return df

        def several_rag_answer(
            is_raw_answer: bool,
            is_vector_only_answer: bool,
            is_graph_only_answer: bool,
            is_graph_vector_answer: bool,
            graph_ratio: float,
            rerank_method: Literal["bleu", "reranker"],
            near_neighbor_first: bool,
            custom_related_information: str,
            answer_prompt: str,
            progress=gr.Progress(track_tqdm=True),
            answer_max_line_count: int = 1,
        ):
            df = pd.read_excel(questions_path, dtype=str)
            total_rows = len(df)
            for index, row in df.iterrows():
                question = row.iloc[0]
                basic_llm_answer, vector_only_answer, graph_only_answer, graph_vector_answer = rag_answer(
                    question,
                    is_raw_answer,
                    is_vector_only_answer,
                    is_graph_only_answer,
                    is_graph_vector_answer,
                    graph_ratio,
                    rerank_method,
                    near_neighbor_first,
                    custom_related_information,
                    answer_prompt,
                )
                df.at[index, "Basic LLM Answer"] = basic_llm_answer
                df.at[index, "Vector-only Answer"] = vector_only_answer
                df.at[index, "Graph-only Answer"] = graph_only_answer
                df.at[index, "Graph-Vector Answer"] = graph_vector_answer
                progress((index + 1, total_rows))
            answers_path = os.path.join(resource_path, "demo", "questions_answers.xlsx")
            df.to_excel(answers_path, index=False)
            return df.head(answer_max_line_count), answers_path

        with gr.Row():
            with gr.Column():
                questions_file = gr.File(file_types=[".xlsx", ".csv"], label="Questions File (.xlsx & csv)")
            with gr.Column():
                test_template_file = os.path.join(resource_path, "demo", "questions_template.xlsx")
                gr.File(value=test_template_file, label="Download Template File")
                answer_max_line_count = gr.Number(1, label="Max Lines To Show", minimum=1, maximum=40)
                answers_btn = gr.Button("Generate Answer (Batch)", variant="primary")
        # TODO: Set individual progress bars for dataframe
        qa_dataframe = gr.DataFrame(label="Questions & Answers (Preview)", headers=tests_df_headers)
        answers_btn.click(
            several_rag_answer,
            inputs=[
                raw_radio,
                vector_only_radio,
                graph_only_radio,
                graph_vector_radio,
                graph_ratio,
                rerank_method,
                near_neighbor_first,
                custom_related_information,
                answer_prompt_input,
                answer_max_line_count,
            ],
            outputs=[qa_dataframe, gr.File(label="Download Answered File", min_width=40)],
        )
        questions_file.change(read_file_to_excel, questions_file, [qa_dataframe, answer_max_line_count])
        answer_max_line_count.change(change_showing_excel, answer_max_line_count, qa_dataframe)

        gr.Markdown("""## 4. Others (🚧) """)
        with gr.Row():
                inp = gr.Textbox(value="g.V().limit(10)", label="Gremlin query", show_copy_button=True, lines=8)
                out = gr.Code(label="Output", language="json", elem_classes="code-container-show")
        btn = gr.Button("Run Gremlin query")
        btn.click(fn=run_gremlin_query, inputs=[inp], outputs=out)  # pylint: disable=no-member

        gr.Markdown("---")
        with gr.Accordion("Init HugeGraph test data (🚧)", open=False):
            with gr.Row():
                inp = []
                out = gr.Textbox(label="Init Graph Demo Result", show_copy_button=True)
            btn = gr.Button("(BETA) Init HugeGraph test data (🚧)")
            btn.click(fn=init_hg_test_data, inputs=inp, outputs=out)  # pylint: disable=no-member
    return hugegraph_llm_ui


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host")
    parser.add_argument("--port", type=int, default=8001, help="port")
    args = parser.parse_args()
    app = FastAPI()
    api_auth = APIRouter(dependencies=[Depends(authenticate)])

    hugegraph_llm = init_rag_ui()
    rag_http_api(api_auth, rag_answer, apply_graph_config, apply_llm_config, apply_embedding_config,
                 apply_reranker_config)

    app.include_router(api_auth)
    auth_enabled = os.getenv("ENABLE_LOGIN", "False").lower() == "true"
    log.info("(Status) Authentication is %s now.", "enabled" if auth_enabled else "disabled")
    # TODO: support multi-user login when need

    app = gr.mount_gradio_app(app, hugegraph_llm, path="/", auth=("rag", os.getenv("TOKEN")) if auth_enabled else None)

    # TODO: we can't use reload now due to the config 'app' of uvicorn.run
    # ❎:f'{__name__}:app' / rag_web_demo:app / hugegraph_llm.demo.rag_web_demo:app
    # TODO: merge unicorn log to avoid duplicate log output (should be unified/fixed later)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
