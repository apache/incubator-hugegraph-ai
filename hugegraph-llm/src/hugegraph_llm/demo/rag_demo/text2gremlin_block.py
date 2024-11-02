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

import gradio as gr

from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.gremlin_generate_task import GremlinGenerator


def build_example_vector_index(temp_file):
    full_path = temp_file.name
    if full_path.endswith(".json"):
        with open(full_path, "r", encoding="utf-8") as f:
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


def create_text2gremlin_block() -> list:
    gr.Markdown("""## 4. Text2gremlin Tools """)

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
