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

import gradio as gr

from hugegraph_llm.utils.hugegraph_utils import init_hg_test_data, run_gremlin_query


def create_other_block():
    gr.Markdown("""## 5. Other Tools """)
    with gr.Row():
        inp = gr.Textbox(value="g.V().limit(10)", label="Gremlin query", show_copy_button=True, lines=8)
        out = gr.Code(label="Output", language="json", elem_classes="code-container-show")
    btn = gr.Button("Run Gremlin query")
    btn.click(fn=run_gremlin_query, inputs=[inp], outputs=out)  # pylint: disable=no-member

    gr.Markdown("---")
    with gr.Accordion("Init HugeGraph test data (🚧)", open=False):
        with gr.Row():
            inp = []
            out = gr.Textbox(label="Init Graph Demo Result", show_copy_button=True)
        btn = gr.Button("(BETA) Init HugeGraph test data (🚧)")
        btn.click(fn=init_hg_test_data, inputs=inp, outputs=out)  # pylint: disable=no-member
