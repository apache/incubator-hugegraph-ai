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

import gradio as gr

from hugegraph_llm.config import prompt
from hugegraph_llm.utils.graph_index_utils import (
    get_graph_index_info,
    clean_all_graph_index,
    update_vid_embedding,
    extract_graph,
    import_graph_data,
)
from hugegraph_llm.utils.vector_index_utils import clean_vector_index, build_vector_index, get_vector_index_info

def store_prompt(doc, schema, example_prompt):
    # update env variables: doc, schema and example_prompt
    if prompt.doc_input_text != doc or prompt.graph_schema != schema or prompt.extract_graph_prompt != example_prompt:
        prompt.doc_input_text = doc
        prompt.graph_schema = schema
        prompt.extract_graph_prompt = example_prompt
        prompt.update_yaml_file()


def create_vector_graph_block():
    # pylint: disable=no-member
    # pylint: disable=C0301
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
                    label="Doc(s)",
                    lines=20,
                    show_copy_button=True
                )
            with gr.Tab("file") as tab_upload_file:
                input_file = gr.File(
                    value=None,
                    label="Docs (multi-files can be selected together)",
                    file_count="multiple",
                )
        input_schema = gr.Textbox(value=prompt.graph_schema, label="Schema", lines=15, show_copy_button=True)
        info_extract_template = gr.Textbox(
            value=prompt.extract_graph_prompt, label="Graph Extract Prompt Header", lines=15, show_copy_button=True
        )
        out = gr.Code(label="Output", language="json", elem_classes="code-container-edit")

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
        graph_extract_bt = gr.Button("Extract Graph Data (1)", variant="primary")
        graph_loading_bt = gr.Button("Load into GraphDB (2)", interactive=True)
        graph_index_rebuild_bt = gr.Button("Update Vid Embedding")

    vector_index_btn0.click(get_vector_index_info, outputs=out).then(
        store_prompt,
        inputs=[input_text, input_schema, info_extract_template],
    )
    vector_index_btn1.click(clean_vector_index).then(
        store_prompt,
        inputs=[input_text, input_schema, info_extract_template],
    )
    vector_import_bt.click(build_vector_index, inputs=[input_file, input_text], outputs=out).then(
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
    graph_index_rebuild_bt.click(update_vid_embedding, outputs=out).then(
        store_prompt,
        inputs=[input_text, input_schema, info_extract_template],
    )

    # origin_out = gr.Textbox(visible=False)
    graph_extract_bt.click(
        extract_graph, inputs=[input_file, input_text, input_schema, info_extract_template], outputs=[out]
    ).then(store_prompt, inputs=[input_text, input_schema, info_extract_template], )

    graph_loading_bt.click(import_graph_data, inputs=[out, input_schema], outputs=[out]).then(update_vid_embedding).then(
        store_prompt,
        inputs=[input_text, input_schema, info_extract_template],
    )

    def on_tab_select(input_f, input_t, evt: gr.SelectData):
        print(f"You selected {evt.value} at {evt.index} from {evt.target}")
        if evt.value == "file":
            return input_f, ""
        if evt.value == "text":
            return [], input_t
        return [], ""

    tab_upload_file.select(fn=on_tab_select, inputs=[input_file, input_text], outputs=[input_file, input_text])
    tab_upload_text.select(fn=on_tab_select, inputs=[input_file, input_text], outputs=[input_file, input_text])

    return input_text, input_schema, info_extract_template
