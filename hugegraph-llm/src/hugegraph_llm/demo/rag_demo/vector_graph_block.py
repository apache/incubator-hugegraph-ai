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

# pylint: disable=E1101

import asyncio
import json
import os

import gradio as gr

from hugegraph_llm.config import huge_settings
from hugegraph_llm.config import prompt
from hugegraph_llm.config import resource_path
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.llm_op.prompt_generate import PromptGenerate
from hugegraph_llm.utils.graph_index_utils import (
    get_graph_index_info,
    clean_all_graph_index,
    clean_all_graph_data,
    update_vid_embedding,
    extract_graph,
    import_graph_data,
    build_schema,
)
from hugegraph_llm.utils.hugegraph_utils import check_graph_db_connection
from hugegraph_llm.utils.log import log
from hugegraph_llm.utils.vector_index_utils import (
    clean_vector_index,
    build_vector_index,
    get_vector_index_info,
)


def store_prompt(doc, schema, example_prompt):
    # update env variables: doc, schema and example_prompt
    if (
        prompt.doc_input_text != doc
        or prompt.graph_schema != schema
        or prompt.extract_graph_prompt != example_prompt
    ):
        prompt.doc_input_text = doc
        prompt.graph_schema = schema
        prompt.extract_graph_prompt = example_prompt
        prompt.update_yaml_file()


def generate_prompt_for_ui(source_text, scenario, example_name):
    """
    Handles the UI logic for generating a new prompt. It calls the PromptGenerate operator.
    """
    if not all([source_text, scenario, example_name]):
        gr.Warning(
            "Please provide original text, expected scenario, and select an example!"
        )
        return gr.update()
    try:
        prompt_generator = PromptGenerate(llm=LLMs().get_chat_llm())
        context = {
            "source_text": source_text,
            "scenario": scenario,
            "example_name": example_name,
        }
        result_context = prompt_generator.run(context)
        # Presents the result of generating prompt
        generated_prompt = result_context.get(
            "generated_extract_prompt", "Generation failed. Please check the logs."
        )
        gr.Info("Prompt generated successfully!")
        return generated_prompt
    except Exception as e:
        log.error("Error generating Prompt: %s", e, exc_info=True)
        raise gr.Error(f"Error generating Prompt: {e}") from e


def load_example_names():
    """Load all candidate examples"""
    try:
        examples_path = os.path.join(
            resource_path, "prompt_examples", "prompt_examples.json"
        )
        with open(examples_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        return [example.get("name", "Unnamed example") for example in examples]
    except (FileNotFoundError, json.JSONDecodeError):
        return ["No available examples"]


def load_query_examples():
    """Load query examples from JSON file based on the prompt language setting"""
    try:
        language = getattr(
            prompt,
            "language",
            getattr(prompt.llm_settings, "language", "EN")
            if hasattr(prompt, "llm_settings")
            else "EN",
        )
        if language.upper() == "CN":
            examples_path = os.path.join(
                resource_path, "prompt_examples", "query_examples_CN.json"
            )
        else:
            examples_path = os.path.join(
                resource_path, "prompt_examples", "query_examples.json"
            )

        with open(examples_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        return json.dumps(examples, indent=2, ensure_ascii=False)
    except (FileNotFoundError, json.JSONDecodeError):
        try:
            examples_path = os.path.join(
                resource_path, "prompt_examples", "query_examples.json"
            )
            with open(examples_path, "r", encoding="utf-8") as f:
                examples = json.load(f)
            return json.dumps(examples, indent=2, ensure_ascii=False)
        except (FileNotFoundError, json.JSONDecodeError):
            return "[]"


def load_schema_fewshot_examples():
    """Load few-shot examples from a JSON file"""
    try:
        examples_path = os.path.join(
            resource_path, "prompt_examples", "schema_examples.json"
        )
        with open(examples_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        return json.dumps(examples, indent=2, ensure_ascii=False)
    except (FileNotFoundError, json.JSONDecodeError):
        return "[]"


def update_example_preview(example_name):
    """Update the display content based on the selected example name."""
    try:
        examples_path = os.path.join(
            resource_path, "prompt_examples", "prompt_examples.json"
        )
        with open(examples_path, "r", encoding="utf-8") as f:
            all_examples = json.load(f)
        selected_example = next(
            (ex for ex in all_examples if ex.get("name") == example_name), None
        )

        if selected_example:
            return (
                selected_example.get("description", ""),
                selected_example.get("text", ""),
                selected_example.get("prompt", ""),
            )
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.warning("Could not update example preview: %s", e)
    return "", "", ""


def _create_prompt_helper_block(demo, input_text, info_extract_template):
    with gr.Accordion("Graph Extraction Prompt Generator", open=False):
        gr.Markdown(
            "Provide your **original text** and **expected scenario**, "
            "then select a reference example to generate a high-quality graph extraction prompt."
        )
        user_scenario_text = gr.Textbox(
            label="Expected scenario/direction",
            info="For example: social relationships, financial knowledge graphs, etc.",
            lines=2,
        )
        example_names = load_example_names()
        few_shot_dropdown = gr.Dropdown(
            choices=example_names,
            label="Select a Few-shot example as a reference",
            value=example_names[0]
            if example_names and example_names[0] != "No available examples"
            else None,
        )
        with gr.Accordion("View example details", open=False):
            example_desc_preview = gr.Markdown(label="Example description")
            example_text_preview = gr.Textbox(
                label="Example input text", lines=5, interactive=False
            )
            example_prompt_preview = gr.Code(
                label="Example Graph Extract Prompt",
                language="markdown",
                interactive=False,
            )

        generate_prompt_btn = gr.Button(
            "🚀 Auto-generate Graph Extract Prompt", variant="primary"
        )
        # Bind the change event of the dropdown menu
        few_shot_dropdown.change(
            fn=update_example_preview,
            inputs=[few_shot_dropdown],
            outputs=[
                example_desc_preview,
                example_text_preview,
                example_prompt_preview,
            ],
        )
        # Bind the click event of the generated button.
        generate_prompt_btn.click(
            fn=generate_prompt_for_ui,
            inputs=[input_text, user_scenario_text, few_shot_dropdown],
            outputs=[info_extract_template],
        )

        # Preload the page on the first load.
        def warm_up_preview(example_name):
            if not example_name:
                return "", "", ""
            return update_example_preview(example_name)

        demo.load(
            fn=warm_up_preview,
            inputs=[few_shot_dropdown],
            outputs=[
                example_desc_preview,
                example_text_preview,
                example_prompt_preview,
            ],
        )


def _build_schema_and_provide_feedback(input_text, query_example, few_shot):
    gr.Info("Generating schema, please wait...")
    # Call the original build_schema function
    generated_schema = build_schema(input_text, query_example, few_shot)
    gr.Info("Schema generated successfully!")
    return generated_schema


def create_vector_graph_block():
    # pylint: disable=no-member
    # pylint: disable=C0301
    # pylint: disable=unexpected-keyword-arg
    with gr.Blocks() as demo:
        gr.Markdown(
            """## Build Vector/Graph Index & Extract Knowledge Graph
    - Docs:
        - text: Build rag index from plain text
        - file: Upload file(s) which should be <u>TXT</u> or <u>.docx</u> (Multiple files can be selected together)
    - [Schema](https://hugegraph.apache.org/docs/clients/restful-api/schema/): (Accept **2 types**)
        - User-defined Schema (JSON format, follow the [template](https://github.com/apache/incubator-hugegraph-ai/blob/aff3bbe25fa91c3414947a196131be812c20ef11/hugegraph-llm/src/hugegraph_llm/config/config_data.py#L125)
        to modify it)
        - Specify the name of the HugeGraph graph instance, it will automatically get the schema from it (like
        **"hugegraph"**)
    - Graph Extract Prompt Header: The user-defined prompt of graph extracting
    - If already exist the graph data, you should click "**Rebuild vid Index**" to update the index
    """
        )

        with gr.Row():
            with gr.Column():
                with gr.Tab("text") as tab_upload_text:
                    input_text = gr.Textbox(
                        value=prompt.doc_input_text,
                        label="Input Doc(s)",
                        lines=20,
                        show_copy_button=True,
                    )
                with gr.Tab("file") as tab_upload_file:
                    input_file = gr.File(
                        value=None,
                        label="Docs (multi-files can be selected together)",
                        file_count="multiple",
                    )
            input_schema = gr.Code(
                value=prompt.graph_schema,
                label="Graph Schema",
                language="json",
                lines=15,
                max_lines=29,
            )
            info_extract_template = gr.Code(
                value=prompt.extract_graph_prompt,
                label="Graph Extract Prompt Header",
                language="markdown",
                lines=15,
                max_lines=29,
            )
            out = gr.Code(
                label="Output Info", language="json", elem_classes="code-container-edit"
            )

        with gr.Row():
            with gr.Accordion("Get RAG Info", open=False):
                with gr.Column():
                    vector_index_btn0 = gr.Button("Get Vector Index Info", size="sm")
                    graph_index_btn0 = gr.Button("Get Graph Index Info", size="sm")
            with gr.Accordion("Clear RAG Data", open=False):
                with gr.Column():
                    vector_index_btn1 = gr.Button(
                        "Clear Chunks Vector Index", size="sm"
                    )
                    graph_index_btn1 = gr.Button(
                        "Clear Graph Vid Vector Index", size="sm"
                    )
                    graph_data_btn0 = gr.Button("Clear Graph Data", size="sm")

            vector_import_bt = gr.Button("Import into Vector", variant="primary")
            graph_extract_bt = gr.Button("Extract Graph Data (1)", variant="primary")
            graph_loading_bt = gr.Button("Load into GraphDB (2)", interactive=True)
            graph_index_rebuild_bt = gr.Button("Update Vid Embedding")

        gr.Markdown("---")
        with gr.Accordion("Graph Schema Generator", open=False):
            gr.Markdown(
                "Provide **query examples** and **few-shot examples**, "
                "then click **Generate Schema** to automatically create graph schema."
            )
            with gr.Row():
                query_example = gr.Code(
                    value=load_query_examples(),
                    label="Query Examples",
                    language="json",
                    lines=10,
                    max_lines=15,
                )
                few_shot = gr.Code(
                    value=load_schema_fewshot_examples(),
                    label="Few-shot Example",
                    language="json",
                    lines=10,
                    max_lines=15,
                )
                build_schema_bt = gr.Button("Generate Schema", variant="primary")
        _create_prompt_helper_block(demo, input_text, info_extract_template)

        vector_index_btn0.click(get_vector_index_info, outputs=out).then(
            store_prompt,
            inputs=[input_text, input_schema, info_extract_template],
        )
        vector_index_btn1.click(clean_vector_index).then(
            store_prompt,
            inputs=[input_text, input_schema, info_extract_template],
        )
        vector_import_bt.click(
            build_vector_index, inputs=[input_file, input_text], outputs=out
        ).then(
            store_prompt,
            inputs=[input_text, input_schema, info_extract_template],
        )
        graph_index_btn0.click(get_graph_index_info, outputs=out).then(
            store_prompt,
            inputs=[input_text, input_schema, info_extract_template],
        )
        graph_index_btn1.click(clean_all_graph_index).then(
            store_prompt,
            inputs=[input_text, input_schema, info_extract_template],
        )
        graph_data_btn0.click(clean_all_graph_data).then(
            store_prompt,
            inputs=[input_text, input_schema, info_extract_template],
        )
        graph_index_rebuild_bt.click(update_vid_embedding, outputs=out).then(
            store_prompt,
            inputs=[input_text, input_schema, info_extract_template],
        )

        # origin_out = gr.Textbox(visible=False)
        graph_extract_bt.click(
            extract_graph,
            inputs=[input_file, input_text, input_schema, info_extract_template],
            outputs=[out],
        ).then(
            store_prompt,
            inputs=[input_text, input_schema, info_extract_template],
        )

        graph_loading_bt.click(
            import_graph_data, inputs=[out, input_schema], outputs=[out]
        ).then(update_vid_embedding).then(
            store_prompt,
            inputs=[input_text, input_schema, info_extract_template],
        )

        # TODO: we should store the examples after the user changed them.
        build_schema_bt.click(
            _build_schema_and_provide_feedback,
            inputs=[input_text, query_example, few_shot],
            outputs=[input_schema],
        ).then(
            store_prompt,
            inputs=[
                input_text,
                input_schema,
                info_extract_template,
            ],  # TODO: Store the updated examples
        )

        def on_tab_select(input_f, input_t, evt: gr.SelectData):
            print(f"You selected {evt.value} at {evt.index} from {evt.target}")
            if evt.value == "file":
                return input_f, ""
            if evt.value == "text":
                return [], input_t
            return [], ""

        tab_upload_file.select(
            fn=on_tab_select,
            inputs=[input_file, input_text],
            outputs=[input_file, input_text],
        )
        tab_upload_text.select(
            fn=on_tab_select,
            inputs=[input_file, input_text],
            outputs=[input_file, input_text],
        )

    return input_text, input_schema, info_extract_template


async def timely_update_vid_embedding(interval_seconds: int = 3600):
    """
    Periodically updates vertex embeddings in the graph database.

    Args:
        :param interval_seconds: Time interval between updates in seconds (default: 3600s -> 1h)
    """
    while True:
        try:
            # Get the latest configuration values on each iteration
            config = {
                "url": huge_settings.graph_url,
                "name": huge_settings.graph_name,
                "user": huge_settings.graph_user,
                "pwd": huge_settings.graph_pwd,
                "graph_space": huge_settings.graph_space,
            }
            if check_graph_db_connection(**config):
                await asyncio.to_thread(update_vid_embedding)
                log.info("update_vid_embedding executed successfully")
            else:
                log.warning(
                    "HugeGraph server connection failed, so skipping update_vid_embedding, "
                    "please check graph configuration and connectivity"
                )
        except asyncio.CancelledError as ce:
            log.info("Periodic task has been cancelled due to: %s", ce)
            break
        # TODO: Add Gradio Warning here
        # pylint: disable=W0718
        except Exception as e:
            log.warning("Failed to execute update_vid_embedding: %s", e, exc_info=True)
        await asyncio.sleep(interval_seconds)
