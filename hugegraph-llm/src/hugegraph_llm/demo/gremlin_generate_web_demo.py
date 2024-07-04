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
import uvicorn
import gradio as gr
from fastapi import FastAPI
from hugegraph_llm.config import settings
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.operators.gremlin_generate_task import GremlinGenerator


def build_example_vector_index(temp_file):
    if str(temp_file).endswith(".json"):
        with open(str(temp_file), "r", encoding="utf-8") as f:
            examples = json.load(f)
    else:
        return "ERROR: please input json file."
    builder = GremlinGenerator(
        llm=LLMs().get_llm(),
        embedding=Embeddings().get_embedding(),
    )
    return builder.example_index_build(examples).run()


def gremlin_generate(inp, use_schema, use_example, example_num, schema):
    generator = GremlinGenerator(
        llm=LLMs().get_llm(),
        embedding=Embeddings().get_embedding(),
    )
    if use_example == "true":
        generator.example_index_query(inp, example_num)
    context = generator.gremlin_generate(use_schema, use_example, schema).run()
    return context.get("match_result", "No Results"), context["result"]


if __name__ == '__main__':
    app = FastAPI()
    with gr.Blocks() as demo:
        gr.Markdown(
            """# HugeGraph LLM Text2Gremlin Demo"""
        )
        gr.Markdown("## Set up the LLM")
        llm_dropdown = gr.Dropdown(["openai", "qianfan_wenxin", "ollama"], value=settings.llm_type,
                                   label="LLM")


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

            def apply_configuration(arg1, arg2, arg3, arg4):
                llm_type = settings.llm_type
                if llm_type == "openai":
                    settings.openai_api_key = arg1
                    settings.openai_api_base = arg2
                    settings.openai_language_model = arg3
                    settings.openai_max_tokens = arg4
                elif llm_type == "qianfan_wenxin":
                    settings.qianfan_api_key = arg1
                    settings.qianfan_secret_key = arg2
                    settings.qianfan_chat_url = arg3
                    settings.qianfan_chat_name = arg4
                elif llm_type == "ollama":
                    settings.ollama_host = arg1
                    settings.ollama_port = int(arg2)
                    settings.ollama_language_model = arg3
                gr.Info("configured!")

            llm_config_button.click(apply_configuration, inputs=llm_config_input)  # pylint: disable=no-member

        gr.Markdown("## Set up the Embedding")
        embedding_dropdown = gr.Dropdown(
            choices=["openai", "ollama"],
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
            else:
                embedding_config_input = []
            embedding_config_button = gr.Button("apply configuration")

            def apply_configuration(arg1, arg2, arg3):
                embedding_type = settings.embedding_type
                if embedding_type == "openai":
                    settings.openai_api_key = arg1
                    settings.openai_api_base = arg2
                    settings.openai_embedding_model = arg3
                elif embedding_type == "ollama":
                    settings.ollama_host = arg1
                    settings.ollama_port = int(arg2)
                    settings.ollama_embedding_model = arg3
                gr.Info("configured!")

            embedding_config_button.click(apply_configuration, inputs=embedding_config_input)  # pylint: disable=no-member

        gr.Markdown("## Build Example Vector Index")
        gr.Markdown("Uploaded json file should be in format below:\n\n"
                    "[{\"query\":\"who is peter\", \"gremlin\":\"g.V().has('name', 'peter')\"}]")
        with gr.Row():
            file = gr.File(label="Upload Example Query-Gremlin Pairs Json")
            out = gr.Textbox(label="Result Message")
        with gr.Row():
            btn = gr.Button("Build Example Vector Index")
        btn.click(build_example_vector_index, inputs=[file], outputs=[out])  # pylint: disable=no-member
        gr.Markdown("## Nature Language To Gremlin")
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
            with gr.Column(scale=1):
                schema_box = gr.Textbox(value=SCHEMA, label="Schema")
            with gr.Column(scale=1):
                input_box = gr.Textbox(value="Tell me about Al Pacino.",
                                       label="Nature Language Query")
                match = gr.Textbox(label="Best-Matched Examples")
                out = gr.Textbox(label="Structured Query Language: Gremlin")
            with gr.Column(scale=1):
                use_example_radio = gr.Radio(choices=["true", "false"], value="false",
                                             label="Use example")
                use_schema_radio = gr.Radio(choices=["true", "false"], value="false",
                                            label="Use schema")
                example_num_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=5,
                    label="Number of examples"
                )
                btn = gr.Button("Text2Gremlin")
        btn.click(  # pylint: disable=no-member
            fn=gremlin_generate,
            inputs=[input_box, use_schema_radio, use_example_radio, example_num_slider, schema_box],
            outputs=[match, out]
        )
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=8002)
