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
from typing import Any, Dict, Literal, Tuple, Union

import gradio as gr
import pandas as pd

from hugegraph_llm.config import huge_settings, index_settings, prompt, resource_path
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.graph_rag_task import RAGPipeline
from hugegraph_llm.operators.gremlin_generate_task import GremlinGenerator
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager
from hugegraph_llm.utils.hugegraph_utils import run_gremlin_query
from hugegraph_llm.utils.log import log
from hugegraph_llm.utils.vector_index_utils import get_vector_index_class


def store_schema(schema, question, gremlin_prompt):
    if (
        prompt.text2gql_graph_schema != schema
        or prompt.default_question != question
        or prompt.gremlin_generate_prompt != gremlin_prompt
    ):
        prompt.text2gql_graph_schema = schema
        prompt.default_question = question
        prompt.gremlin_generate_prompt = gremlin_prompt
        prompt.update_yaml_file()


def build_example_vector_index(temp_file) -> dict:
    vector_index = get_vector_index_class(index_settings.now_vector_index)
    assert vector_index, 'vector db name is error'
    if temp_file is None:
        full_path = os.path.join(resource_path, "demo", "text2gremlin.csv")
    else:
        full_path = temp_file.name
    if full_path.endswith(".json"):
        with open(full_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
    elif full_path.endswith(".csv"):
        examples = pd.read_csv(full_path).to_dict("records")
    else:
        log.critical("Unsupported file format. Please input a JSON or CSV file.")
        return {"error": "Unsupported file format. Please input a JSON or CSV file."}
    builder = GremlinGenerator(
        llm=LLMs().get_text2gql_llm(),
        embedding=Embeddings().get_embedding(),
    )
    return builder.example_index_build(examples, vector_index=vector_index).run()


def gremlin_generate(
    inp, example_num, schema, gremlin_prompt
) -> Union[tuple[str, str], tuple[str, Any, Any, Any, Any]]:
    vector_index = get_vector_index_class(index_settings.now_vector_index)
    generator = GremlinGenerator(llm=LLMs().get_text2gql_llm(), embedding=Embeddings().get_embedding())
    sm = SchemaManager(graph_name=schema)
    short_schema = False

    if schema:
        schema = schema.strip()
        if not schema.startswith("{"):
            short_schema = True
            log.info("Try to get schema from graph '%s'", schema)
            generator.import_schema(from_hugegraph=schema)
            # FIXME: update the logic here
            schema = sm.schema.getSchema()
        else:
            try:
                schema = json.loads(schema)
                generator.import_schema(from_user_defined=schema)
            except json.JSONDecodeError as e:
                log.error("Invalid JSON schema provided: %s", e)
                return "Invalid JSON schema, please check the format carefully.", ""
    # FIXME: schema is not used in gremlin_generate() step, no context for it (enhance the logic here)
    updated_schema = sm.simple_schema(schema) if short_schema else schema
    store_schema(str(updated_schema), inp, gremlin_prompt)
    context = (
        generator.example_index_query(example_num, vector_index)
        .gremlin_generate_synthesize(updated_schema, gremlin_prompt)
        .run(query=inp)
    )
    try:
        context["template_exec_res"] = run_gremlin_query(query=context["result"])
    except Exception as e:  # pylint: disable=broad-except
        context["template_exec_res"] = f"{e}"
    try:
        context["raw_exec_res"] = run_gremlin_query(query=context["raw_result"])
    except Exception as e:  # pylint: disable=broad-except
        context["raw_exec_res"] = f"{e}"

    match_result = json.dumps(context.get("match_result", "No Results"), ensure_ascii=False, indent=2)
    return (
        match_result,
        context["result"],
        context["raw_result"],
        context["template_exec_res"],
        context["raw_exec_res"],
    )


def simple_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    mini_schema = {}  # type: ignore

    # Add necessary vertexlabels items (3)
    if "vertexlabels" in schema:
        mini_schema["vertexlabels"] = []
        for vertex in schema["vertexlabels"]:
            new_vertex = {key: vertex[key] for key in ["id", "name", "properties"] if key in vertex}
            mini_schema["vertexlabels"].append(new_vertex)

    # Add necessary edgelabels items (4)
    if "edgelabels" in schema:
        mini_schema["edgelabels"] = []
        for edge in schema["edgelabels"]:
            new_edge = {key: edge[key] for key in ["name", "source_label", "target_label", "properties"] if key in edge}
            mini_schema["edgelabels"].append(new_edge)

    return mini_schema


def create_text2gremlin_block() -> Tuple:
    gr.Markdown(
        """## Build Vector Template Index (Optional)
    > Uploaded CSV file should be in `query,gremlin` format below:    
    > e.g. `who is peter?`,`g.V().has('name', 'peter')`    
    > JSON file should be in format below:  
    > e.g. `[{"query":"who is peter", "gremlin":"g.V().has('name', 'peter')"}]`
    """
    )
    with gr.Row():
        file = gr.File(
            value=os.path.join(resource_path, "demo", "text2gremlin.csv"), label="Upload Text-Gremlin Pairs File"
        )
        out = gr.Textbox(label="Result Message")
    with gr.Row():
        btn = gr.Button("Build Example Vector Index", variant="primary")

    btn.click(build_example_vector_index, inputs=[file], outputs=[out])  # pylint: disable=no-member
    gr.Markdown("## Nature Language To Gremlin")

    with gr.Row():
        with gr.Column(scale=1):
            input_box = gr.Textbox(value=prompt.default_question, label="Nature Language Query", show_copy_button=True)
            match = gr.Code(label="Similar Template (TopN)", language="javascript", elem_classes="code-container-show")
            initialized_out = gr.Textbox(label="Gremlin With Template", show_copy_button=True)
            raw_out = gr.Textbox(label="Gremlin Without Template", show_copy_button=True)
            tmpl_exec_out = gr.Code(
                label="Query With Template Output", language="json", elem_classes="code-container-show"
            )
            raw_exec_out = gr.Code(
                label="Query Without Template Output", language="json", elem_classes="code-container-show"
            )

        with gr.Column(scale=1):
            example_num_slider = gr.Slider(minimum=0, maximum=10, step=1, value=2, label="Number of refer examples")
            schema_box = gr.Textbox(value=prompt.text2gql_graph_schema, label="Schema", lines=2, show_copy_button=True)
            prompt_box = gr.Textbox(
                value=prompt.gremlin_generate_prompt, label="Prompt", lines=20, show_copy_button=True
            )
            btn = gr.Button("Text2Gremlin", variant="primary")
    btn.click(  # pylint: disable=no-member
        fn=gremlin_generate,
        inputs=[input_box, example_num_slider, schema_box, prompt_box],
        outputs=[match, initialized_out, raw_out, tmpl_exec_out, raw_exec_out],
    )

    return input_box, schema_box, prompt_box


def graph_rag_recall(
    query: str,
    gremlin_tmpl_num: int,
    rerank_method: Literal["bleu", "reranker"],
    near_neighbor_first: bool,
    custom_related_information: str,
    gremlin_prompt: str,
    max_graph_items: int,
    topk_return_results: int,
    vector_dis_threshold: float,
    topk_per_keyword: int,
    get_vertex_only: bool = False,
) -> dict:
    store_schema(prompt.text2gql_graph_schema, query, gremlin_prompt)
    rag = RAGPipeline()
    rag.extract_keywords().keywords_to_vid(
        vector_index_str=index_settings.now_vector_index,
        vector_dis_threshold=vector_dis_threshold,
        topk_per_keyword=topk_per_keyword,
    )

    if not get_vertex_only:
        rag.import_schema(huge_settings.graph_name).query_graphdb(
            num_gremlin_generate_example=gremlin_tmpl_num,
            gremlin_prompt=gremlin_prompt,
            max_graph_items=max_graph_items,
        ).merge_dedup_rerank(
            rerank_method=rerank_method,
            near_neighbor_first=near_neighbor_first,
            custom_related_information=custom_related_information,
            topk_return_results=topk_return_results,
        )
    context = rag.run(verbose=True, query=query, graph_search=True)
    return context
