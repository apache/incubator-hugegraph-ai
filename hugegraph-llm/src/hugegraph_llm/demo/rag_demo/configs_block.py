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
from typing import Optional

import gradio as gr
import requests
from requests.auth import HTTPBasicAuth

from hugegraph_llm.config import huge_settings, admin_settings, llm_settings
from hugegraph_llm.utils.log import log
from functools import partial

current_llm = "chat"


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
            except (json.decoder.JSONDecodeError, AttributeError) as e:
                raise gr.Error(resp.text) from e
    return resp.status_code


def config_qianfan_model(arg1, arg2, arg3=None, settings_prefix=None, origin_call=None) -> int:
    setattr(llm_settings, f"qianfan_{settings_prefix}_api_key", arg1)
    setattr(llm_settings, f"qianfan_{settings_prefix}_secret_key", arg2)
    if arg3:
        setattr(llm_settings, f"qianfan_{settings_prefix}_language_model", arg3)
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
    embedding_option = llm_settings.embedding_type
    if embedding_option == "openai":
        llm_settings.openai_embedding_api_key = arg1
        llm_settings.openai_embedding_api_base = arg2
        llm_settings.openai_embedding_model = arg3
        test_url = llm_settings.openai_embedding_api_base + "/embeddings"
        headers = {"Authorization": f"Bearer {arg1}"}
        data = {"model": arg3, "input": "test"}
        status_code = test_api_connection(test_url, method="POST", headers=headers, body=data, origin_call=origin_call)
    elif embedding_option == "qianfan_wenxin":
        status_code = config_qianfan_model(arg1, arg2, settings_prefix="embedding", origin_call=origin_call)
        llm_settings.qianfan_embedding_model = arg3
    elif embedding_option == "ollama/local":
        llm_settings.ollama_embedding_host = arg1
        llm_settings.ollama_embedding_port = int(arg2)
        llm_settings.ollama_embedding_model = arg3
        status_code = test_api_connection(f"http://{arg1}:{arg2}", origin_call=origin_call)
    llm_settings.update_env()
    gr.Info("Configured!")
    return status_code


def apply_reranker_config(
    reranker_api_key: Optional[str] = None,
    reranker_model: Optional[str] = None,
    cohere_base_url: Optional[str] = None,
    origin_call=None,
) -> int:
    status_code = -1
    reranker_option = llm_settings.reranker_type
    if reranker_option == "cohere":
        llm_settings.reranker_api_key = reranker_api_key
        llm_settings.reranker_model = reranker_model
        llm_settings.cohere_base_url = cohere_base_url
        headers = {"Authorization": f"Bearer {reranker_api_key}"}
        status_code = test_api_connection(
            cohere_base_url.rsplit("/", 1)[0] + "/check-api-key",
            method="POST",
            headers=headers,
            origin_call=origin_call,
        )
    elif reranker_option == "siliconflow":
        llm_settings.reranker_api_key = reranker_api_key
        llm_settings.reranker_model = reranker_model
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
    llm_settings.update_env()
    gr.Info("Configured!")
    return status_code


def apply_graph_config(ip, port, name, user, pwd, gs, origin_call=None) -> int:
    huge_settings.graph_ip = ip
    huge_settings.graph_port = port
    huge_settings.graph_name = name
    huge_settings.graph_user = user
    huge_settings.graph_pwd = pwd
    huge_settings.graph_space = gs
    # Test graph connection (Auth)
    if gs and gs.strip():
        test_url = f"http://{ip}:{port}/graphspaces/{gs}/graphs/{name}/schema"
    else:
        test_url = f"http://{ip}:{port}/graphs/{name}/schema"
    auth = HTTPBasicAuth(user, pwd)
    # for http api return status
    response = test_api_connection(test_url, auth=auth, origin_call=origin_call)
    huge_settings.update_env()
    return response


# Different llm models have different parameters, so no meaningful argument names are given here
def apply_llm_config(current_llm, arg1, arg2, arg3, arg4, origin_call=None) -> int:
    log.debug("current llm in apply_llm_config is %s", current_llm)
    llm_option = getattr(llm_settings, f"{current_llm}_llm_type")
    log.debug("llm option in apply_llm_config is %s", llm_option)
    status_code = -1
    
    if llm_option == "openai":
        setattr(llm_settings, f"openai_{current_llm}_api_key", arg1)
        setattr(llm_settings, f"openai_{current_llm}_api_base", arg2)
        setattr(llm_settings, f"openai_{current_llm}_language_model", arg3)
        setattr(llm_settings, f"openai_{current_llm}_tokens", int(arg4))
        
        test_url = getattr(llm_settings, f"openai_{current_llm}_api_base") + "/chat/completions"
        log.debug(f"Type of openai {current_llm} max token is %s", type(arg4))
        data = {
            "model": arg3,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "test"}],
        }
        headers = {"Authorization": f"Bearer {arg1}"}
        status_code = test_api_connection(test_url, method="POST", headers=headers, body=data, origin_call=origin_call)
    
    elif llm_option == "qianfan_wenxin":
        status_code = config_qianfan_model(arg1, arg2, arg3, settings_prefix=current_llm, origin_call=origin_call)
    
    elif llm_option == "ollama/local":
        setattr(llm_settings, f"ollama_{current_llm}_host", arg1)
        setattr(llm_settings, f"ollama_{current_llm}_port", int(arg2))
        setattr(llm_settings, f"ollama_{current_llm}_language_model", arg3)
        status_code = test_api_connection(f"http://{arg1}:{arg2}", origin_call=origin_call)

    gr.Info("Configured!")
    llm_settings.update_env()
    
    return status_code


# TODO: refactor the function to reduce the number of statements & separate the logic
def create_configs_block() -> list:
    # pylint: disable=R0915 (too-many-statements)
    with gr.Accordion("1. Set up the HugeGraph server.", open=False):
        with gr.Row():
            graph_config_input = [
                gr.Textbox(value=huge_settings.graph_ip, label="ip"),
                gr.Textbox(value=huge_settings.graph_port, label="port"),
                gr.Textbox(value=huge_settings.graph_name, label="graph"),
                gr.Textbox(value=huge_settings.graph_user, label="user"),
                gr.Textbox(value=huge_settings.graph_pwd, label="pwd", type="password"),
                gr.Textbox(value=huge_settings.graph_space, label="graphspace(Optional)"),
            ]
        graph_config_button = gr.Button("Apply Configuration")
    graph_config_button.click(apply_graph_config, inputs=graph_config_input)  # pylint: disable=no-member

    #TODO : use OOP to restruact
    with gr.Accordion("2. Set up the LLM.", open=False):
        gr.Markdown("> Tips: the openai option also support openai style api from other providers.")
        with gr.Tab(label='chat'):
            chat_llm_dropdown = gr.Dropdown(choices=["openai", "qianfan_wenxin", "ollama/local"],
                            value=getattr(llm_settings, f"chat_llm_type"), label=f"type")
            apply_llm_config_with_chat_op = partial(apply_llm_config, "chat")
            @gr.render(inputs=[chat_llm_dropdown])
            def chat_llm_settings(llm_type):
                llm_settings.chat_llm_type = llm_type
                llm_config_input = []
                if llm_type == "openai":
                    llm_config_input = [
                        gr.Textbox(value=getattr(llm_settings, f"openai_chat_api_key"), label="api_key", type="password"),
                        gr.Textbox(value=getattr(llm_settings, f"openai_chat_api_base"), label="api_base"),
                        gr.Textbox(value=getattr(llm_settings, f"openai_chat_language_model"), label="model_name"),
                        gr.Textbox(value=getattr(llm_settings, f"openai_chat_tokens"), label="max_token"),
                ]
                elif llm_type == "ollama/local":
                    llm_config_input = [
                        gr.Textbox(value=getattr(llm_settings, f"ollama_chat_host"), label="host"),
                        gr.Textbox(value=str(getattr(llm_settings, f"ollama_chat_port")), label="port"),
                        gr.Textbox(value=getattr(llm_settings, f"ollama_chat_language_model"), label="model_name"),
                        gr.Textbox(value="", visible=False),
                    ]
                elif llm_type == "qianfan_wenxin":
                    llm_config_input = [
                        gr.Textbox(value=getattr(llm_settings, f"qianfan_chat_api_key"), label="api_key", type="password"),
                        gr.Textbox(value=getattr(llm_settings, f"qianfan_chat_secret_key"), label="secret_key", type="password"),
                        gr.Textbox(value=getattr(llm_settings, f"qianfan_chat_language_model"), label="model_name"),
                        gr.Textbox(value="", visible=False),
                    ]
                else:
                    llm_config_input = [gr.Textbox(value="", visible=False) for _ in range(4)]
                llm_config_button = gr.Button("Apply configuration")
                llm_config_button.click(apply_llm_config_with_chat_op, inputs=llm_config_input)

        with gr.Tab(label='mini_tasks'):
            extract_llm_dropdown = gr.Dropdown(choices=["openai", "qianfan_wenxin", "ollama/local"],
                        value=getattr(llm_settings, f"extract_llm_type"), label=f"type")
            apply_llm_config_with_extract_op = partial(apply_llm_config, "extract")

            @gr.render(inputs=[extract_llm_dropdown])
            def extract_llm_settings(llm_type):
                llm_settings.extract_llm_type = llm_type
                llm_config_input = []
                if llm_type == "openai":
                    llm_config_input = [
                        gr.Textbox(value=getattr(llm_settings, f"openai_extract_api_key"), label="api_key", type="password"),
                        gr.Textbox(value=getattr(llm_settings, f"openai_extract_api_base"), label="api_base"),
                        gr.Textbox(value=getattr(llm_settings, f"openai_extract_language_model"), label="model_name"),
                        gr.Textbox(value=getattr(llm_settings, f"openai_extract_tokens"), label="max_token"),
                ]
                elif llm_type == "ollama/local":
                    llm_config_input = [
                        gr.Textbox(value=getattr(llm_settings, f"ollama_extract_host"), label="host"),
                        gr.Textbox(value=str(getattr(llm_settings, f"ollama_extract_port")), label="port"),
                        gr.Textbox(value=getattr(llm_settings, f"ollama_extract_language_model"), label="model_name"),
                        gr.Textbox(value="", visible=False),
                    ]
                elif llm_type == "qianfan_wenxin":
                    llm_config_input = [
                        gr.Textbox(value=getattr(llm_settings, f"qianfan_extract_api_key"), label="api_key", type="password"),
                        gr.Textbox(value=getattr(llm_settings, f"qianfan_extract_secret_key"), label="secret_key", type="password"),
                        gr.Textbox(value=getattr(llm_settings, f"qianfan_extract_language_model"), label="model_name"),
                        gr.Textbox(value="", visible=False),
                    ]
                else:
                    llm_config_input = [gr.Textbox(value="", visible=False) for _ in range(4)]
                llm_config_button = gr.Button("Apply configuration")
                llm_config_button.click(apply_llm_config_with_extract_op, inputs=llm_config_input)
        with gr.Tab(label='text2gql'):
            text2gql_llm_dropdown = gr.Dropdown(choices=["openai", "qianfan_wenxin", "ollama/local"],
                            value=getattr(llm_settings, f"text2gql_llm_type"), label=f"type")
            apply_llm_config_with_text2gql_op = partial(apply_llm_config, "text2gql")

            @gr.render(inputs=[text2gql_llm_dropdown])
            def text2gql_llm_settings(llm_type):
                llm_settings.text2gql_llm_type = llm_type
                llm_config_input = []
                if llm_type == "openai":
                    llm_config_input = [
                        gr.Textbox(value=getattr(llm_settings, f"openai_text2gql_api_key"), label="api_key", type="password"),
                        gr.Textbox(value=getattr(llm_settings, f"openai_text2gql_api_base"), label="api_base"),
                        gr.Textbox(value=getattr(llm_settings, f"openai_text2gql_language_model"), label="model_name"),
                        gr.Textbox(value=getattr(llm_settings, f"openai_text2gql_tokens"), label="max_token"),
                    ]
                elif llm_type == "ollama/local":
                    llm_config_input = [
                        gr.Textbox(value=getattr(llm_settings, f"ollama_text2gql_host"), label="host"),
                        gr.Textbox(value=str(getattr(llm_settings, f"ollama_text2gql_port")), label="port"),
                        gr.Textbox(value=getattr(llm_settings, f"ollama_text2gql_language_model"), label="model_name"),
                        gr.Textbox(value="", visible=False),
                    ]
                elif llm_type == "qianfan_wenxin":
                    llm_config_input = [
                        gr.Textbox(value=getattr(llm_settings, f"qianfan_text2gql_api_key"), label="api_key", type="password"),
                        gr.Textbox(value=getattr(llm_settings, f"qianfan_text2gql_secret_key"), label="secret_key", type="password"),
                        gr.Textbox(value=getattr(llm_settings, f"qianfan_text2gql_language_model"), label="model_name"),
                        gr.Textbox(value="", visible=False),
                    ]
                else:
                    llm_config_input = [gr.Textbox(value="", visible=False) for _ in range(4)]
                llm_config_button = gr.Button("Apply configuration")
                llm_config_button.click(apply_llm_config_with_text2gql_op, inputs=llm_config_input)


    with gr.Accordion("3. Set up the Embedding.", open=False):
        embedding_dropdown = gr.Dropdown(
            choices=["openai", "qianfan_wenxin", "ollama/local"], value=llm_settings.embedding_type, label="Embedding"
        )

        @gr.render(inputs=[embedding_dropdown])
        def embedding_settings(embedding_type):
            llm_settings.embedding_type = embedding_type
            if embedding_type == "openai":
                with gr.Row():
                    embedding_config_input = [
                        gr.Textbox(value=llm_settings.openai_embedding_api_key, label="api_key", type="password"),
                        gr.Textbox(value=llm_settings.openai_embedding_api_base, label="api_base"),
                        gr.Textbox(value=llm_settings.openai_embedding_model, label="model_name"),
                    ]
            elif embedding_type == "ollama/local":
                with gr.Row():
                    embedding_config_input = [
                        gr.Textbox(value=llm_settings.ollama_embedding_host, label="host"),
                        gr.Textbox(value=str(llm_settings.ollama_embedding_port), label="port"),
                        gr.Textbox(value=llm_settings.ollama_embedding_model, label="model_name"),
                    ]
            elif embedding_type == "qianfan_wenxin":
                with gr.Row():
                    embedding_config_input = [
                        gr.Textbox(value=llm_settings.qianfan_embedding_api_key, label="api_key", type="password"),
                        gr.Textbox(value=llm_settings.qianfan_embedding_secret_key, label="secret_key", type="password"),
                        gr.Textbox(value=llm_settings.qianfan_embedding_model, label="model_name"),
                    ]
            else:
                embedding_config_input = [
                    gr.Textbox(value="", visible=False),
                    gr.Textbox(value="", visible=False),
                    gr.Textbox(value="", visible=False),
                ]

            embedding_config_button = gr.Button("Apply Configuration")

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
            llm_settings.reranker_type = reranker_type if reranker_type != "None" else None
            if reranker_type == "cohere":
                with gr.Row():
                    reranker_config_input = [
                        gr.Textbox(value=llm_settings.reranker_api_key, label="api_key", type="password"),
                        gr.Textbox(value=llm_settings.reranker_model, label="model"),
                        gr.Textbox(value=llm_settings.cohere_base_url, label="base_url"),
                    ]
            elif reranker_type == "siliconflow":
                with gr.Row():
                    reranker_config_input = [
                        gr.Textbox(value=llm_settings.reranker_api_key, label="api_key", type="password"),
                        gr.Textbox(
                            value="BAAI/bge-reranker-v2-m3",
                            label="model",
                            info="Please refer to https://siliconflow.cn/pricing",
                        ),
                        gr.Textbox(value="", visible=False),
                    ]
            else:
                reranker_config_input = [
                    gr.Textbox(value="", visible=False),
                    gr.Textbox(value="", visible=False),
                    gr.Textbox(value="", visible=False),
                ]
            reranker_config_button = gr.Button("Apply configuration")

            # TODO: use "gr.update()" or other way to update the config in time (refactor the click event)
            # Call the separate apply_reranker_configuration function here
            reranker_config_button.click(  # pylint: disable=no-member
                fn=apply_reranker_config,
                inputs=reranker_config_input,  # pylint: disable=no-member
            )
    # The reason for returning this partial value is the functional need to refresh the ui
    return graph_config_input
