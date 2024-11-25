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

import pandas as pd
import gradio as gr

from hugegraph_llm.config import prompt
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.gremlin_generate_task import GremlinGenerator
from hugegraph_llm.utils.log import log


def build_example_vector_index(temp_file):
    full_path = temp_file.name
    if full_path.endswith(".json"):
        with open(full_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
    elif full_path.endswith(".csv"):
        examples = pd.read_csv(full_path).to_dict('records')
    else:
        return "ERROR: please input json file."
    builder = GremlinGenerator(
        llm=LLMs().get_text2gql_llm(),
        embedding=Embeddings().get_embedding(),
    )
    return builder.example_index_build(examples).run()


def gremlin_generate(inp, example_num, schema):
    generator = GremlinGenerator(
        llm=LLMs().get_text2gql_llm(),
        embedding=Embeddings().get_embedding(),
    )
    if schema:
        try:
            schema = json.loads(schema.strip())
            generator.import_schema(from_user_defined=schema)
        except json.JSONDecodeError:
            log.info("Get schema from graph!")
            generator.import_schema(from_hugegraph=schema)
    context = generator.example_index_query(example_num).gremlin_generate(schema).run(query=inp)
    return context.get("match_result", "No Results"), context["result"]


def create_text2gremlin_block() -> list:
    gr.Markdown("""## Text2gremlin Tools """)

    gr.Markdown("## Build Example Vector Index")
    gr.Markdown("Uploaded json file should be in format below:  \n"
                "[{\"query\":\"who is peter\", \"gremlin\":\"g.V().has('name', 'peter')\"}]  \n"
                "Uploaded csv file should be in format below:  \n"
                "query,gremlin  \n\"who is peter\",\"g.V().has('name', 'peter')\"")
    with gr.Row():
        file = gr.File(label="Upload Example Query-Gremlin Pairs Json")
        out = gr.Textbox(label="Result Message")
    with gr.Row():
        btn = gr.Button("Build Example Vector Index")
    btn.click(build_example_vector_index, inputs=[file], outputs=[out])  # pylint: disable=no-member
    gr.Markdown("## Nature Language To Gremlin")

    with gr.Row():
        with gr.Column(scale=1):
            schema_box = gr.Textbox(value=prompt.graph_schema, label="Schema", lines=10)
        with gr.Column(scale=1):
            input_box = gr.Textbox(value="Tell me about Al Pacino.",
                                   label="Nature Language Query")
            match = gr.Textbox(label="Best-Matched Examples")
            out = gr.Textbox(label="Structured Query Language: Gremlin")
        with gr.Column(scale=1):
            example_num_slider = gr.Slider(
                minimum=0,
                maximum=10,
                step=1,
                value=5,
                label="Number of examples"
            )
            btn = gr.Button("Text2Gremlin")
    btn.click(  # pylint: disable=no-member
        fn=gremlin_generate,
        inputs=[input_box, example_num_slider, schema_box],
        outputs=[match, out]
    )
