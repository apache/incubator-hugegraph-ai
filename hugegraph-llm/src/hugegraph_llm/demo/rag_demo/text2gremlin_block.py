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
from typing import Any

import gradio as gr
import pandas as pd

from hugegraph_llm.config import prompt, resource_path
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.gremlin_generate_task import GremlinGenerator
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager
from hugegraph_llm.utils.hugegraph_utils import run_gremlin_query

from hugegraph_llm.utils.log import log


def store_schema(schema, question):
    if prompt.text2gql_graph_schema != schema or prompt.default_question != question:
        prompt.text2gql_graph_schema = schema
        prompt.default_question = question
        prompt.update_yaml_file()

def build_example_vector_index(temp_file) -> dict:
    if temp_file == None:
        full_path = os.path.join(resource_path, "demo", "text2gremlin.csv")
    else:
        full_path = temp_file.name
    if full_path.endswith(".json"):
        with open(full_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
    elif full_path.endswith(".csv"):
        examples = pd.read_csv(full_path).to_dict('records')
    else:
        log.critical("Unsupported file format. Please input a JSON or CSV file.")
        return {"error": "Unsupported file format. Please input a JSON or CSV file."}
    builder = GremlinGenerator(
        llm=LLMs().get_text2gql_llm(),
        embedding=Embeddings().get_embedding(),
    )
    return builder.example_index_build(examples).run()


def gremlin_generate(inp, example_num, schema) -> tuple[str, str] | tuple[str, Any, Any, Any, Any]:
    generator = GremlinGenerator(llm=LLMs().get_text2gql_llm(), embedding=Embeddings().get_embedding())
    if schema:
        schema = schema.strip()
        if not schema.startswith("{"):
            log.info("Try to get schema from graph '%s'", schema)
            generator.import_schema(from_hugegraph=schema)
        else:
            try:
                schema = json.loads(schema)
                generator.import_schema(from_user_defined=schema)
            except json.JSONDecodeError as e:
                log.error("Invalid JSON schema provided: %s", e)
                return "Invalid JSON schema, please check the format carefully.", ""
    # FIXME: schema is not used in gremlin_generate() step, no context for it (enhance the logic here)
    updated_schema = SchemaManager(graph_name=schema).schema.getSchema()
    context = generator.example_index_query(example_num).gremlin_generate_synthesize(updated_schema).run(query=inp)
    try:
        context["template_exec_res"] = run_gremlin_query(query=context["result"])
    except Exception as e:
        context["template_exec_res"] = f"{e}"
    try:
        context["raw_exec_res"] = run_gremlin_query(query=context["raw_result"])
    except Exception as e:
        context["raw_exec_res"] = f"{e}"

    match_result = json.dumps(context.get("match_result", "No Results"), ensure_ascii=False, indent=2)
    return match_result, context["result"], context["raw_result"], context["template_exec_res"], context["raw_exec_res"]


def create_text2gremlin_block():
    gr.Markdown("""## Build Vector Template Index (Optional)  
    > Uploaded CSV file should be in `query,gremlin` format below:    
    > e.g. `who is peter?`,`g.V().has('name', 'peter')`    
    > JSON file should be in format below:  
    > e.g. `[{"query":"who is peter", "gremlin":"g.V().has('name', 'peter')"}]`
    """)
    with gr.Row():
        file = gr.File(
            value=os.path.join(resource_path, "demo", "text2gremlin.csv"),
            label="Upload Text-Gremlin Pairs File"
        )
        out = gr.Textbox(label="Result Message")
    with gr.Row():
        btn = gr.Button("Build Example Vector Index", variant="primary")
    btn.click(build_example_vector_index, inputs=[file], outputs=[out])  # pylint: disable=no-member
    gr.Markdown("## Nature Language To Gremlin")

    with gr.Row():
        with gr.Column(scale=1):
            input_box = gr.Textbox(value=prompt.default_question, label="Nature Language Query")
            match = gr.Code(label="Best-Matched Examples", language="javascript", elem_classes="code-container-show")
            initialized_out = gr.Textbox(label="Gremlin With Template", show_copy_button=True)
            raw_out = gr.Textbox(label="Gremlin Without Template", show_copy_button=True)
            tmpl_exec_out = gr.Code(label="Query With Template Output", language="json",
                                    elem_classes="code-container-show")
            raw_exec_out = gr.Code(label="Query Without Template Output", language="json",
                                   elem_classes="code-container-show")

        with gr.Column(scale=1):
            example_num_slider = gr.Slider(
                minimum=0,
                maximum=10,
                step=1,
                value=2,
                label="Number of refer examples"
            )
            schema_box = gr.Textbox(value=prompt.text2gql_graph_schema, label="Schema", lines=2)
            btn = gr.Button("Text2Gremlin", variant="primary")
    btn.click(  # pylint: disable=no-member
        fn=gremlin_generate,
        inputs=[input_box, example_num_slider, schema_box],
        outputs=[match, initialized_out, raw_out, tmpl_exec_out, raw_exec_out]
    ).then(store_schema, inputs=[schema_box, input_box],)
