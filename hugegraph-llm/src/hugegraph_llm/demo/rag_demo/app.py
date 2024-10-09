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
import os

import gradio as gr
import uvicorn
from fastapi import FastAPI, Depends, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from hugegraph_llm.api.rag_api import rag_http_api
from hugegraph_llm.demo.rag_demo.configs_block import (
    create_configs_block,
    apply_llm_config,
    apply_embedding_config,
    apply_reranker_config,
    apply_graph_config,
)
from hugegraph_llm.demo.rag_demo.other_block import create_other_block
from hugegraph_llm.demo.rag_demo.rag_block import create_rag_block, rag_answer
from hugegraph_llm.demo.rag_demo.vector_graph_block import create_vector_graph_block
from hugegraph_llm.resources.demo.css import CSS
from hugegraph_llm.utils.log import log
from hugegraph_llm.config import settings, prompt

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


def init_rag_ui() -> gr.Interface:
    with gr.Blocks(
        theme="default",
        title="HugeGraph RAG Platform",
        css=CSS,
    ) as hugegraph_llm_ui:
        gr.Markdown("# HugeGraph LLM RAG Demo")

        # create_configs_block()

        # graph_config_input = textbox_array_graph_config
        #  = [settings.graph_ip, settings.graph_port, settings.graph_name, graph_user, settings.graph_pwd, settings.graph_space]
        
        # llm_config_input = textbox_array_llm_config
        # 判断 settings.llm_type，但是这个值似乎也缺少一个刷新机制
        #  = if settings.llm_type == openai [settings.openai_api_key, settings.openai_api_base, settings.openai_language_model, settings.openai_max_tokens]
        #  = else if settings.llm_type == ollama [settings.ollama_host, settings.ollama_port, settings.ollama_language_model, ""]
        #  = else if settings.llm_type == qianfan_wenxin [settings.qianfan_api_key, settings.qianfan_secret_key, settings.qianfan_language_model, ""]
        #  = else ["","","", ""] 这个是错误情况

        # embedding_config_input = textbox_array_embedding_config
        # 判断 settings.embedding_type，但是这个值也缺少一个刷新机制
        #  = if settings.embedding_type == openai [settings.openai_api_key, settings.openai_api_base, settings.openai_embedding_model]
        #  = else if settings.embedding_type == ollama [settings.ollama_host, settings.ollama_port, settings.ollama_embedding_model]
        #  = else if settings.embedding_type == qianfan_wenxin [settings.qianfan_api_key, settings.qianfan_secret_key, settings.qianfan_embedding_model]
        #  = else ["","",""]

        # reranker_config_input = textbox_array_reranker_config
        # 判断 settings.reranker_type，但是这个值也缺少一个刷新机制
        #  = if settings.reranker_type == cohere [settings.reranker_api_key, settings.reranker_model, settings.cohere_base_url]
        #  = else if settings.reranker_type == siliconflow [settings.reranker_api_key, "BAAI/bge-reranker-v2-m3", ""]
        #  = else ["","",""]

        textbox_array_graph_config, textbox_array_llm_config, textbox_array_embedding_config, textbox_array_reranker_config = create_configs_block()

        with gr.Tab(label="1. Build RAG Index 💡"):
            create_vector_graph_block()
        with gr.Tab(label="2. (Graph)RAG & User Functions 📖"):
            create_rag_block()
        with gr.Tab(label="3. Others Tools 🚧"):
            create_other_block()
        
        def refresh_ui_config_prompt():
            
            settings.from_env()

            if settings.llm_type == "openai":
                llm_config_arg_0 = settings.openai_api_key
                llm_config_arg_1 = settings.openai_api_base
                llm_config_arg_2 = settings.openai_language_model
                llm_config_arg_3 = settings.openai_max_tokens
            elif settings.llm_type == "ollama":
                llm_config_arg_0 = settings.ollama_host
                llm_config_arg_1 = settings.ollama_port
                llm_config_arg_2 = settings.ollama_language_model
                llm_config_arg_3 = ""
            elif settings.llm_type == "qianfan_wenxin":
                llm_config_arg_0 = settings.qianfan_api_key
                llm_config_arg_1 = settings.qianfan_secret_key
                llm_config_arg_2 = settings.qianfan_language_model
                llm_config_arg_3 = ""
            else:
                llm_config_arg_0 = ""
                llm_config_arg_1 = ""
                llm_config_arg_2 = ""
                llm_config_arg_3 = ""

            if settings.embedding_type == "openai":
                embedding_config_arg_0 = settings.openai_api_key
                embedding_config_arg_1 = settings.openai_api_base
                embedding_config_arg_2 = settings.openai_embedding_model
            elif settings.embedding_type == "ollama":
                embedding_config_arg_0 = settings.ollama_host
                embedding_config_arg_1 = settings.ollama_port
                embedding_config_arg_2 = settings.ollama_embedding_model
            elif settings.embedding_type == "qianfan_wenxin":
                embedding_config_arg_0 = settings.qianfan_api_key
                embedding_config_arg_1 = settings.qianfan_secret_key
                embedding_config_arg_2 = settings.qianfan_embedding_model
            else:
                embedding_config_arg_0 = ""
                embedding_config_arg_1 = ""
                embedding_config_arg_2 = ""

            if settings.reranker_type == "cohere":
                reranker_config_arg_0 = settings.reranker_api_key
                reranker_config_arg_1 = settings.reranker_model
                reranker_config_arg_2 = settings.cohere_base_url
            elif settings.reranker_type == "siliconflow":
                reranker_config_arg_0 = settings.reranker_api_key
                reranker_config_arg_1 = "BAAI/bge-reranker-v2-m3"
                reranker_config_arg_2 = ""
            else:
                reranker_config_arg_0 = ""
                reranker_config_arg_1 = ""
                reranker_config_arg_2 = ""

            return settings.graph_ip, settings.graph_port, settings.graph_name, settings.graph_user, settings.graph_pwd, settings.graph_space,llm_config_arg_0, llm_config_arg_1, llm_config_arg_2, llm_config_arg_3, embedding_config_arg_0, embedding_config_arg_1, embedding_config_arg_2, reranker_config_arg_0, reranker_config_arg_1, reranker_config_arg_2

        hugegraph_llm_ui.load(fn=refresh_ui_config_prompt, outputs=[
            textbox_array_graph_config[0],
            textbox_array_graph_config[1],
            textbox_array_graph_config[2],
            textbox_array_graph_config[3],
            textbox_array_graph_config[4],
            textbox_array_graph_config[5],

            textbox_array_llm_config[0],
            textbox_array_llm_config[1],
            textbox_array_llm_config[2],
            textbox_array_llm_config[3],

            textbox_array_embedding_config[0],
            textbox_array_embedding_config[1],
            textbox_array_embedding_config[2],

            textbox_array_reranker_config[0],
            textbox_array_reranker_config[1],
            textbox_array_reranker_config[2],
        ])

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
