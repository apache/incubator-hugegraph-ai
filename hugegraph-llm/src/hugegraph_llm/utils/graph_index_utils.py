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


import os
import traceback
from typing import Dict, Any, Union, List

import gradio as gr
from hugegraph_llm.flows.scheduler import SchedulerSingleton
from pyhugegraph.client import PyHugeClient

from .embedding_utils import get_filename_prefix, get_index_folder_name
from .hugegraph_utils import clean_hg_data
from .log import log
from .vector_index_utils import read_documents
from ..config import resource_path, huge_settings, llm_settings
from ..indices.vector_index.faiss_vector_store import FaissVectorIndex
from ..models.embeddings.init_embedding import Embeddings


def get_graph_index_info():
    try:
        scheduler = SchedulerSingleton.get_instance()
        return scheduler.schedule_flow("get_graph_index_info")
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error(e)
        raise gr.Error(str(e))


def clean_all_graph_index():
    folder_name = get_index_folder_name(
        huge_settings.graph_name, huge_settings.graph_space
    )
    filename_prefix = get_filename_prefix(
        llm_settings.embedding_type,
        getattr(Embeddings().get_embedding(), "model_name", None),
    )
    FaissVectorIndex.clean(
        str(os.path.join(resource_path, folder_name, "graph_vids")), filename_prefix
    )
    FaissVectorIndex.clean(
        str(os.path.join(resource_path, folder_name, "gremlin_examples")),
        filename_prefix,
    )
    log.warning("Clear graph index and text2gql index successfully!")
    gr.Info("Clear graph index and text2gql index successfully!")


def get_vertex_details(
    vertex_ids: List[str], context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    if isinstance(context.get("graph_client"), PyHugeClient):
        client = context["graph_client"]
    else:
        url = context.get("url") or "http://localhost:8080"
        graph = context.get("graph") or "hugegraph"
        user = context.get("user") or "admin"
        pwd = context.get("pwd") or "admin"
        gs = context.get("graphspace") or None
        client = PyHugeClient(url, graph, user, pwd, gs)
    if not vertex_ids:
        return []

    formatted_ids = ", ".join(f"'{vid}'" for vid in vertex_ids)
    gremlin_query = f"g.V({formatted_ids}).limit(20)"
    result = client.gremlin().exec(gremlin=gremlin_query)["data"]
    return result


def clean_all_graph_data():
    clean_hg_data()
    log.warning("Clear graph data successfully!")
    gr.Info("Clear graph data successfully!")


def extract_graph(input_file, input_text, schema, example_prompt) -> str:
    texts = read_documents(input_file, input_text)
    scheduler = SchedulerSingleton.get_instance()
    if not schema:
        return "ERROR: please input with correct schema/format."

    try:
        return scheduler.schedule_flow(
            "graph_extract", schema, texts, example_prompt, "property_graph"
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error(e)
        raise gr.Error(str(e))


def update_vid_embedding():
    scheduler = SchedulerSingleton.get_instance()
    try:
        return scheduler.schedule_flow("update_vid_embeddings")
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error(e)
        raise gr.Error(str(e))


def import_graph_data(data: str, schema: str) -> Union[str, Dict[str, Any]]:
    try:
        scheduler = SchedulerSingleton.get_instance()
        return scheduler.schedule_flow("import_graph_data", data, schema)
    except Exception as e:  # pylint: disable=W0718
        log.error(e)
        traceback.print_exc()
        # Note: can't use gr.Error here
        gr.Warning(str(e) + " Please check the graph data format/type carefully.")
        return data


def build_schema(input_text, query_example, few_shot):
    scheduler = SchedulerSingleton.get_instance()
    try:
        return scheduler.schedule_flow(
            "build_schema", input_text, query_example, few_shot
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error("Schema generation failed: %s", e)
        raise gr.Error(f"Schema generation failed: {e}")
