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

import docx
import gradio as gr

from hugegraph_llm.config import resource_path, huge_settings
from hugegraph_llm.indices.vector_index import VectorIndex
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.kg_construction_task import KgBuilder
from hugegraph_llm.utils.hugegraph_utils import get_hg_client


def read_documents(input_file, input_text):
    if input_text:
        texts = [input_text]
    elif input_file:
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
    else:
        raise gr.Error("Please input text or upload file.")
    return texts


#pylint: disable=C0301
def get_vector_index_info():
    chunk_vector_index = VectorIndex.from_index_file(str(os.path.join(resource_path, huge_settings.graph_name, "chunks")))
    graph_vid_vector_index = VectorIndex.from_index_file(str(os.path.join(resource_path, huge_settings.graph_name, "graph_vids")))
    graph_prop_vector_index = VectorIndex.from_index_file(str(os.path.join(resource_path, huge_settings.graph_name, "graph_props")))
    return json.dumps({
        "embed_dim": chunk_vector_index.index.d,
        "vector_info": {
            "chunk_vector_num": chunk_vector_index.index.ntotal,
            "graph_vid_vector_num": graph_vid_vector_index.index.ntotal,
            "graph_properties_vector_num": graph_prop_vector_index.index.ntotal,
        }
    }, ensure_ascii=False, indent=2)


def clean_vector_index():
    VectorIndex.clean(str(os.path.join(resource_path, huge_settings.graph_name, "chunks")))
    gr.Info("Clean vector index successfully!")


def build_vector_index(input_file, input_text):
    if input_file and input_text:
        raise gr.Error("Please only choose one between file and text.")
    texts = read_documents(input_file, input_text)
    builder = KgBuilder(LLMs().get_chat_llm(), Embeddings().get_embedding(), get_hg_client())
    context = builder.chunk_split(texts, "paragraph", "zh").build_vector_index().run()
    return json.dumps(context, ensure_ascii=False, indent=2)
