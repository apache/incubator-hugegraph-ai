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
from hugegraph_llm.api.admin_api import admin_http_api
from hugegraph_llm.config import settings, prompt
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
from hugegraph_llm.demo.rag_demo.admin_block import create_admin_block, log_stream
from hugegraph_llm.resources.demo.css import CSS
from hugegraph_llm.utils.log import log

sec = HTTPBearer()


def authenticate(credentials: HTTPAuthorizationCredentials = Depends(sec)):
    correct_token = os.getenv("USER_TOKEN")
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

        """
        TODO: leave a general idea of the unresolved part
        graph_config_input = textbox_array_graph_config
         = [settings.graph_ip, settings.graph_port, settings.graph_name, graph_user, settings.graph_pwd, settings.graph_space]
        
        llm_config_input = textbox_array_llm_config
         = if settings.llm_type == openai [settings.openai_api_key, settings.openai_api_base, settings.openai_language_model, settings.openai_max_tokens]
         = else if settings.llm_type == ollama [settings.ollama_host, settings.ollama_port, settings.ollama_language_model, ""]
         = else if settings.llm_type == qianfan_wenxin [settings.qianfan_api_key, settings.qianfan_secret_key, settings.qianfan_language_model, ""]
         = else ["","","", ""]

        embedding_config_input = textbox_array_embedding_config
         = if settings.embedding_type == openai [settings.openai_api_key, settings.openai_api_base, settings.openai_embedding_model]
         = else if settings.embedding_type == ollama [settings.ollama_host, settings.ollama_port, settings.ollama_embedding_model]
         = else if settings.embedding_type == qianfan_wenxin [settings.qianfan_api_key, settings.qianfan_secret_key, settings.qianfan_embedding_model]
         = else ["","",""]

        reranker_config_input = textbox_array_reranker_config
         = if settings.reranker_type == cohere [settings.reranker_api_key, settings.reranker_model, settings.cohere_base_url]
         = else if settings.reranker_type == siliconflow [settings.reranker_api_key, "BAAI/bge-reranker-v2-m3", ""]
         = else ["","",""]
        """

        textbox_array_graph_config = create_configs_block()

        with gr.Tab(label="1. Build RAG Index 💡"):
            textbox_input_schema, textbox_info_extract_template = create_vector_graph_block()
        with gr.Tab(label="2. (Graph)RAG & User Functions 📖"):
            textbox_inp, textbox_answer_prompt_input = create_rag_block()
        with gr.Tab(label="3. Graph Tools 🚧"):
            create_other_block()
        with gr.Tab(label="4. Admin Tools ⚙️"):
            create_admin_block()
        

        def refresh_ui_config_prompt() -> tuple:
            settings.from_env()
            prompt.ensure_yaml_file_exists()
            return (
                settings.graph_ip, settings.graph_port, settings.graph_name, settings.graph_user,
                settings.graph_pwd, settings.graph_space, prompt.graph_schema, prompt.extract_graph_prompt,
                prompt.default_question, prompt.answer_prompt
            )

        hugegraph_llm_ui.load(fn=refresh_ui_config_prompt, outputs=[
            textbox_array_graph_config[0],
            textbox_array_graph_config[1],
            textbox_array_graph_config[2],
            textbox_array_graph_config[3],
            textbox_array_graph_config[4],
            textbox_array_graph_config[5],

            textbox_input_schema,
            textbox_info_extract_template,

            textbox_inp,
            textbox_answer_prompt_input
        ])

    return hugegraph_llm_ui


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host")
    parser.add_argument("--port", type=int, default=8001, help="port")
    args = parser.parse_args()
    app = FastAPI()
    api_auth = APIRouter(dependencies=[Depends(authenticate)])
    
    settings.check_env()
    prompt.update_yaml_file()

    hugegraph_llm = init_rag_ui()
    rag_http_api(api_auth, rag_answer, apply_graph_config, apply_llm_config, apply_embedding_config,
                 apply_reranker_config)
    admin_http_api(api_auth, log_stream)
    
    app.include_router(api_auth)
    auth_enabled = os.getenv("ENABLE_LOGIN", "False").lower() == "true"
    log.info("(Status) Authentication is %s now.", "enabled" if auth_enabled else "disabled")
    # TODO: support multi-user login when need
    app = gr.mount_gradio_app(app, hugegraph_llm, path="/", auth=("rag", os.getenv("TOKEN")) if auth_enabled else None)

    # TODO: we can't use reload now due to the config 'app' of uvicorn.run
    # ❎:f'{__name__}:app' / rag_web_demo:app / hugegraph_llm.demo.rag_web_demo:app
    # TODO: merge unicorn log to avoid duplicate log output (should be unified/fixed later)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
