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
from .vector_index_utils import read_documents
from ..config import resource_path, settings
from ..indices.vector_index import VectorIndex
from ..models.embeddings.init_embedding import Embeddings
from ..models.llms.init_llm import LLMs
from ..operators.kg_construction_task import KgBuilder


def get_graph_index_info():
    builder = KgBuilder(LLMs().get_llm(), Embeddings().get_embedding(), get_hg_client())
    context = builder.fetch_graph_data().run()
    vector_index = VectorIndex.from_index_file(str(os.path.join(resource_path, settings.graph_name, "graph_vids")))
    context["vid_index"] = {
        "embed_dim": vector_index.index.d,
        "num_vectors": vector_index.index.ntotal,
        "num_vids": len(vector_index.properties)
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def clean_graph_index():
    clean_hg_data()
    VectorIndex.clean(str(os.path.join(resource_path, settings.graph_name, "graph_vids")))
    gr.Info("Clean graph index successfully!")


def extract_graph(input_file, input_text, schema, example_prompt):
    texts = read_documents(input_file, input_text)
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
    (builder
     .chunk_split(texts, "paragraph", "zh")
     .extract_info(example_prompt, "property_graph"))
    log.debug(builder.operators)
    try:
        context = builder.run()
        return (
            f"Extract {len(context['vertices'])} entities and {len(context['edges'])} relations successfully.",
            json.dumps(context, ensure_ascii=False, indent=2),
            gr.Column(visible=True)
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error(e)
        raise gr.Error(str(e))


def fit_vid_index():
    builder = KgBuilder(LLMs().get_llm(), Embeddings().get_embedding(), get_hg_client())
    builder.fetch_graph_data().build_vertex_id_semantic_index()
    log.debug(builder.operators)
    try:
        context = builder.run()
        removed_num = context["removed_vid_vector_num"]
        added_num = context["added_vid_vector_num"]
        return f"Removed {removed_num} vectors, added {added_num} vectors."
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error(e)
        raise gr.Error(str(e))


def build_graph_index(input_file, input_text, schema, example_prompt):
    texts = read_documents(input_file, input_text)
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
    (builder
     .chunk_split(texts, "paragraph", "zh")
     .extract_info(example_prompt, "property_graph")
     .commit_to_hugegraph()
     .build_vertex_id_semantic_index())
    log.debug(builder.operators)
    try:
        context = builder.run()
        return json.dumps(context, ensure_ascii=False, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error(e)
        raise gr.Error(str(e))

def import_graph_data(data: str):
    data = json.loads(data.strip())
    builder = KgBuilder(LLMs().get_llm(), Embeddings().get_embedding(), get_hg_client())
    context = builder.commit_to_hugegraph().run(data)
    return gr.Column(visible=False), context
