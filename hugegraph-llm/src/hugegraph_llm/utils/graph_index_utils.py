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

import gradio as gr
from .hugegraph_utils import get_hg_client, clean_hg_data
from .log import log
from ..config import resource_path
from ..indices.vector_index import VectorIndex
from ..models.embeddings.init_embedding import Embeddings
from ..models.llms.init_llm import LLMs
from ..operators.kg_construction_task import KgBuilder


def get_graph_index_info():
    client = get_hg_client()
    vector_index = VectorIndex.from_index_file(os.path.join(resource_path, "graph_vids"))
    return client.graph_manager.get_graph_index_info()


def clean_graph_index():
    clean_hg_data()
    VectorIndex.clean(str(os.path.join(resource_path, settings.graph_name, "graph_vids")))


def extract_graph_data(input_file, input_text, schema, example_prompt):
    if input_file:
        texts = []
        for file in input_file:
            full_path = file.name
            if full_path.endswith(".txt"):
                with open(full_path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            elif full_path.endswith(".docx"):
                text = ""
                doc = docx.Document(full_path)
                for para in doc.paragraphs:
                    text += para.text
                    text += "\n"
                texts.append(text)
            elif full_path.endswith(".pdf"):
                # TODO: support PDF file
                raise gr.Error("PDF will be supported later! Try to upload text/docx now")
            else:
                raise gr.Error("Please input txt or docx file.")
    elif input_text:
        texts = [input_text]
    else:
        raise gr.Error("Please input text or upload file.")
    builder = KgBuilder(LLMs().get_llm(), Embeddings().get_embedding(), get_hg_client())

    if schema:
        try:
            schema = json.loads(schema.strip())
            builder.import_schema(from_user_defined=schema)
        except json.JSONDecodeError as e:
            log.error(e)
            builder.import_schema(from_hugegraph=schema)
        else:
            return "ERROR: please input schema."
    builder.chunk_split(texts, "paragraph", "zh")

    builder.extract_info(example_prompt, "property_graph")

    if build_mode in (BuildMode.CLEAR_AND_IMPORT.value, BuildMode.IMPORT_MODE.value):
        builder.commit_to_hugegraph()
    if build_mode != BuildMode.TEST_MODE.value:
        builder.build_vertex_id_semantic_index()
    log.debug(builder.operators)
    try:
        context = builder.run()
        return str(context)
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error(e)
        raise gr.Error(str(e))
