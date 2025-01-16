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

import os
from typing import Tuple, Literal, Optional

import gradio as gr
import pandas as pd
from gradio.utils import NamedString

from hugegraph_llm.config import resource_path, prompt, huge_settings, llm_settings
from hugegraph_llm.operators.graph_rag_task import RAGPipeline
from hugegraph_llm.utils.log import log


def rag_answer(
    text: str,
    raw_answer: bool,
    vector_only_answer: bool,
    graph_only_answer: bool,
    graph_vector_answer: bool,
    graph_ratio: float,
    rerank_method: Literal["bleu", "reranker"],
    near_neighbor_first: bool,
    custom_related_information: str,
    answer_prompt: str,
    keywords_extract_prompt: str,
    gremlin_tmpl_num: Optional[int] = 2,
    gremlin_prompt: Optional[str] = None,
) -> Tuple:
    """
    Generate an answer using the RAG (Retrieval-Augmented Generation) pipeline.
    1. Initialize the RAGPipeline.
    2. Select vector search or graph search based on parameters.
    3. Merge, deduplicate, and rerank the results.
    4. Synthesize the final answer.
    5. Run the pipeline and return the results.
    """

    gremlin_prompt = gremlin_prompt or prompt.gremlin_generate_prompt
    should_update_prompt = (
        prompt.default_question != text
        or prompt.answer_prompt != answer_prompt
        or prompt.keywords_extract_prompt != keywords_extract_prompt
        or prompt.gremlin_generate_prompt != gremlin_prompt
        or prompt.custom_rerank_info != custom_related_information
    )
    if should_update_prompt:
        prompt.custom_rerank_info = custom_related_information
        prompt.default_question = text
        prompt.answer_prompt = answer_prompt
        prompt.keywords_extract_prompt = keywords_extract_prompt
        prompt.gremlin_generate_prompt = gremlin_prompt
        prompt.update_yaml_file()

    vector_search = vector_only_answer or graph_vector_answer
    graph_search = graph_only_answer or graph_vector_answer
    if raw_answer is False and not vector_search and not graph_search:
        gr.Warning("Please select at least one generate mode.")
        return "", "", "", ""

    rag = RAGPipeline()
    if vector_search:
        rag.query_vector_index()
    if graph_search:
        rag.extract_keywords(extract_template=keywords_extract_prompt).keywords_to_vid().import_schema(
            huge_settings.graph_name
        ).query_graphdb(
            num_gremlin_generate_example=gremlin_tmpl_num,
            gremlin_prompt=gremlin_prompt,
        )
    # TODO: add more user-defined search strategies
    rag.merge_dedup_rerank(
        graph_ratio,
        rerank_method,
        near_neighbor_first,
    )
    rag.synthesize_answer(raw_answer, vector_only_answer, graph_only_answer, graph_vector_answer, answer_prompt)

    try:
        context = rag.run(verbose=True, query=text, vector_search=vector_search, graph_search=graph_search)
        if context.get("switch_to_bleu"):
            gr.Warning("Online reranker fails, automatically switches to local bleu rerank.")
        return (
            context.get("raw_answer", ""),
            context.get("vector_only_answer", ""),
            context.get("graph_only_answer", ""),
            context.get("graph_vector_answer", ""),
        )
    except ValueError as e:
        log.critical(e)
        raise gr.Error(str(e))
    except Exception as e:
        log.critical(e)
        raise gr.Error(f"An unexpected error occurred: {str(e)}")


def create_rag_block():
    # pylint: disable=R0915 (too-many-statements),C0301
    gr.Markdown("""## 1. HugeGraph RAG Query""")
    with gr.Row():
        with gr.Column(scale=2):
            inp = gr.Textbox(value=prompt.default_question, label="Question", show_copy_button=True, lines=3)

            # TODO: Only support inline formula now. Should support block formula
            gr.Markdown("Basic LLM Answer", elem_classes="output-box-label")
            raw_out = gr.Markdown(elem_classes="output-box", show_copy_button=True, latex_delimiters=[{"left":"$", "right":"$", "display":False}])
            gr.Markdown("Vector-only Answer", elem_classes="output-box-label")
            vector_only_out = gr.Markdown(elem_classes="output-box", show_copy_button=True, latex_delimiters=[{"left":"$", "right":"$", "display":False}])
            gr.Markdown("Graph-only Answer", elem_classes="output-box-label")
            graph_only_out = gr.Markdown(elem_classes="output-box", show_copy_button=True, latex_delimiters=[{"left":"$", "right":"$", "display":False}])
            gr.Markdown("Graph-Vector Answer", elem_classes="output-box-label")
            graph_vector_out = gr.Markdown(elem_classes="output-box", show_copy_button=True, latex_delimiters=[{"left":"$", "right":"$", "display":False}])

            answer_prompt_input = gr.Textbox(
                value=prompt.answer_prompt, label="Query Prompt", show_copy_button=True, lines=7
            )
            keywords_extract_prompt_input = gr.Textbox(
                value=prompt.keywords_extract_prompt,
                label="Keywords Extraction Prompt",
                show_copy_button=True,
                lines=7,
            )
        with gr.Column(scale=1):
            with gr.Row():
                raw_radio = gr.Radio(choices=[True, False], value=False, label="Basic LLM Answer")
                vector_only_radio = gr.Radio(choices=[True, False], value=False, label="Vector-only Answer")
            with gr.Row():
                graph_only_radio = gr.Radio(choices=[True, False], value=True, label="Graph-only Answer")
                graph_vector_radio = gr.Radio(choices=[True, False], value=False, label="Graph-Vector Answer")

            def toggle_slider(enable):
                return gr.update(interactive=enable)

            with gr.Column():
                with gr.Row():
                    online_rerank = llm_settings.reranker_type
                    rerank_method = gr.Dropdown(
                        choices=["bleu", ("rerank (online)", "reranker")] if online_rerank else ["bleu"],
                        value="reranker" if online_rerank else "bleu",
                        label="Rerank method",
                    )
                    example_num = gr.Number(value=2, label="Template Num (0 to disable it) ", precision=0)
                    graph_ratio = gr.Slider(0, 1, 0.6, label="Graph Ratio", step=0.1, interactive=False)

                graph_vector_radio.change(
                    toggle_slider, inputs=graph_vector_radio, outputs=graph_ratio
                )  # pylint: disable=no-member
                near_neighbor_first = gr.Checkbox(
                    value=False,
                    label="Near neighbor first(Optional)",
                    info="One-depth neighbors > two-depth neighbors",
                )
                custom_related_information = gr.Text(
                    prompt.custom_rerank_info,
                    label="Query related information(Optional)",
                )
                btn = gr.Button("Answer Question", variant="primary")

    btn.click(  # pylint: disable=no-member
        fn=rag_answer,
        inputs=[
            inp,
            raw_radio,
            vector_only_radio,
            graph_only_radio,
            graph_vector_radio,
            graph_ratio,
            rerank_method,
            near_neighbor_first,
            custom_related_information,
            answer_prompt_input,
            keywords_extract_prompt_input,
            example_num,
        ],
        outputs=[raw_out, vector_only_out, graph_only_out, graph_vector_out],
    )

    gr.Markdown(
        """## 2. (Batch) Back-testing )
    > 1. Download the template file & fill in the questions you want to test.
    > 2. Upload the file & click the button to generate answers. (Preview shows the first 40 lines)
    > 3. The answer options are the same as the above RAG/Q&A frame 
    """
    )
    tests_df_headers = [
        "Question",
        "Expected Answer",
        "Basic LLM Answer",
        "Vector-only Answer",
        "Graph-only Answer",
        "Graph-Vector Answer",
    ]
    # FIXME: "demo" might conflict with the graph name, it should be modified.
    answers_path = os.path.join(resource_path, "demo", "questions_answers.xlsx")
    questions_path = os.path.join(resource_path, "demo", "questions.xlsx")
    questions_template_path = os.path.join(resource_path, "demo", "questions_template.xlsx")

    def read_file_to_excel(file: NamedString, line_count: Optional[int] = None):
        df = None
        if not file:
            return pd.DataFrame(), 1
        if file.name.endswith(".xlsx"):
            df = pd.read_excel(file.name, nrows=line_count) if file else pd.DataFrame()
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file.name, nrows=line_count) if file else pd.DataFrame()
        df.to_excel(questions_path, index=False)
        if df.empty:
            df = pd.DataFrame([[""] * len(tests_df_headers)], columns=tests_df_headers)
        else:
            df.columns = tests_df_headers
        # truncate the dataframe if it's too long
        if len(df) > 40:
            return df.head(40), 40
        return df, len(df)

    def change_showing_excel(line_count):
        if os.path.exists(answers_path):
            df = pd.read_excel(answers_path, nrows=line_count)
        elif os.path.exists(questions_path):
            df = pd.read_excel(questions_path, nrows=line_count)
        else:
            df = pd.read_excel(questions_template_path, nrows=line_count)
        return df

    def several_rag_answer(
        is_raw_answer: bool,
        is_vector_only_answer: bool,
        is_graph_only_answer: bool,
        is_graph_vector_answer: bool,
        graph_ratio: float,
        rerank_method: Literal["bleu", "reranker"],
        near_neighbor_first: bool,
        custom_related_information: str,
        answer_prompt: str,
        keywords_extract_prompt: str,
        answer_max_line_count: int = 1,
        progress=gr.Progress(track_tqdm=True),
    ):
        df = pd.read_excel(questions_path, dtype=str)
        total_rows = len(df)
        for index, row in df.iterrows():
            question = row.iloc[0]
            basic_llm_answer, vector_only_answer, graph_only_answer, graph_vector_answer = rag_answer(
                question,
                is_raw_answer,
                is_vector_only_answer,
                is_graph_only_answer,
                is_graph_vector_answer,
                graph_ratio,
                rerank_method,
                near_neighbor_first,
                custom_related_information,
                answer_prompt,
                keywords_extract_prompt,
            )
            df.at[index, "Basic LLM Answer"] = basic_llm_answer
            df.at[index, "Vector-only Answer"] = vector_only_answer
            df.at[index, "Graph-only Answer"] = graph_only_answer
            df.at[index, "Graph-Vector Answer"] = graph_vector_answer
            progress((index + 1, total_rows))
        answers_path = os.path.join(resource_path, "demo", "questions_answers.xlsx")
        df.to_excel(answers_path, index=False)
        return df.head(answer_max_line_count), answers_path

    with gr.Row():
        with gr.Column():
            questions_file = gr.File(file_types=[".xlsx", ".csv"], label="Questions File (.xlsx & csv)")
        with gr.Column():
            test_template_file = os.path.join(resource_path, "demo", "questions_template.xlsx")
            gr.File(value=test_template_file, label="Download Template File")
            answer_max_line_count = gr.Number(1, label="Max Lines To Show", minimum=1, maximum=40)
            answers_btn = gr.Button("Generate Answer (Batch)", variant="primary")
    # TODO: Set individual progress bars for dataframe
    qa_dataframe = gr.DataFrame(label="Questions & Answers (Preview)", headers=tests_df_headers)
    answers_btn.click(
        several_rag_answer,
        inputs=[
            raw_radio,
            vector_only_radio,
            graph_only_radio,
            graph_vector_radio,
            graph_ratio,
            rerank_method,
            near_neighbor_first,
            custom_related_information,
            answer_prompt_input,
            keywords_extract_prompt_input,
            answer_max_line_count,
        ],
        outputs=[qa_dataframe, gr.File(label="Download Answered File", min_width=40)],
    )
    questions_file.change(read_file_to_excel, questions_file, [qa_dataframe, answer_max_line_count])
    answer_max_line_count.change(change_showing_excel, answer_max_line_count, qa_dataframe)
    return inp, answer_prompt_input, keywords_extract_prompt_input, custom_related_information
