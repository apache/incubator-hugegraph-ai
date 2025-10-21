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
import traceback
from typing import Dict, Any, Union, Optional

import gradio as gr
from hugegraph_llm.flows.scheduler import SchedulerSingleton

from .embedding_utils import get_filename_prefix, get_index_folder_name
from .hugegraph_utils import get_hg_client, clean_hg_data
from .log import log
from .vector_index_utils import read_documents
from ..config import resource_path, huge_settings, llm_settings
from ..indices.vector_index import VectorIndex
from ..models.embeddings.init_embedding import Embeddings
from ..models.llms.init_llm import LLMs
from ..operators.kg_construction_task import KgBuilder


def get_graph_index_info():
    builder = KgBuilder(
        LLMs().get_chat_llm(), Embeddings().get_embedding(), get_hg_client()
    )
    graph_summary_info = builder.fetch_graph_data().run()
    folder_name = get_index_folder_name(
        huge_settings.graph_name, huge_settings.graph_space
    )
    index_dir = str(os.path.join(resource_path, folder_name, "graph_vids"))
    filename_prefix = get_filename_prefix(
        llm_settings.embedding_type, getattr(builder.embedding, "model_name", None)
    )
    vector_index = VectorIndex.from_index_file(index_dir, filename_prefix)
    graph_summary_info["vid_index"] = {
        "embed_dim": vector_index.index.d,
        "num_vectors": vector_index.index.ntotal,
        "num_vids": len(vector_index.properties),
    }
    return json.dumps(graph_summary_info, ensure_ascii=False, indent=2)


def clean_all_graph_index():
    folder_name = get_index_folder_name(
        huge_settings.graph_name, huge_settings.graph_space
    )
    filename_prefix = get_filename_prefix(
        llm_settings.embedding_type,
        getattr(Embeddings().get_embedding(), "model_name", None),
    )
    VectorIndex.clean(
        str(os.path.join(resource_path, folder_name, "graph_vids")), filename_prefix
    )
    VectorIndex.clean(
        str(os.path.join(resource_path, folder_name, "gremlin_examples")),
        filename_prefix,
    )
    log.warning("Clear graph index and text2gql index successfully!")
    gr.Info("Clear graph index and text2gql index successfully!")


def clean_all_graph_data():
    clean_hg_data()
    log.warning("Clear graph data successfully!")
    gr.Info("Clear graph data successfully!")


def parse_schema(schema: str, builder: KgBuilder) -> Optional[str]:
    schema = schema.strip()
    if schema.startswith("{"):
        try:
            schema = json.loads(schema)
            builder.import_schema(from_user_defined=schema)
        except json.JSONDecodeError:
            log.error("Invalid JSON format in schema. Please check it again.")
            return "ERROR: Invalid JSON format in schema. Please check it carefully."
    else:
        log.info("Get schema '%s' from graphdb.", schema)
        builder.import_schema(from_hugegraph=schema)
    return None


def extract_graph_origin(input_file, input_text, schema, example_prompt) -> str:
    texts = read_documents(input_file, input_text)
    builder = KgBuilder(
        LLMs().get_chat_llm(), Embeddings().get_embedding(), get_hg_client()
    )
    if not schema:
        return "ERROR: please input with correct schema/format."

    error_message = parse_schema(schema, builder)
    if error_message:
        return error_message
    builder.chunk_split(texts, "document", "zh").extract_info(
        example_prompt, "property_graph"
    )

    try:
        context = builder.run()
        if not context["vertices"] and not context["edges"]:
            log.info("Please check the schema.(The schema may not match the Doc)")
            return json.dumps(
                {
                    "vertices": context["vertices"],
                    "edges": context["edges"],
                    "warning": "The schema may not match the Doc",
                },
                ensure_ascii=False,
                indent=2,
            )
        return json.dumps(
            {"vertices": context["vertices"], "edges": context["edges"]},
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error(e)
        raise gr.Error(str(e))


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
    builder = KgBuilder(
        LLMs().get_chat_llm(), Embeddings().get_embedding(), get_hg_client()
    )
    builder.fetch_graph_data().build_vertex_id_semantic_index()
    log.debug("Operators: %s", builder.operators)
    try:
        context = builder.run()
        removed_num = context["removed_vid_vector_num"]
        added_num = context["added_vid_vector_num"]
        return f"Removed {removed_num} vectors, added {added_num} vectors."
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.error(e)
        raise gr.Error(str(e))


def import_graph_data(data: str, schema: str) -> Union[str, Dict[str, Any]]:
    try:
        data_json = json.loads(data.strip())
        log.debug("Import graph data: %s", data)
        builder = KgBuilder(
            LLMs().get_chat_llm(), Embeddings().get_embedding(), get_hg_client()
        )
        if schema:
            error_message = parse_schema(schema, builder)
            if error_message:
                return error_message

        context = builder.commit_to_hugegraph().run(data_json)
        gr.Info("Import graph data successfully!")
        print(context)
        return json.dumps(context, ensure_ascii=False, indent=2)
    except Exception as e:  # pylint: disable=W0718
        log.error(e)
        traceback.print_exc()
        # Note: can't use gr.Error here
        gr.Warning(str(e) + " Please check the graph data format/type carefully.")
        return data


def build_schema(input_text, query_example, few_shot):
    context = {
        "raw_texts": [input_text] if input_text else [],
        "query_examples": [],
        "few_shot_schema": {},
    }

    if few_shot:
        try:
            context["few_shot_schema"] = json.loads(few_shot)
        except json.JSONDecodeError as e:
            raise gr.Error(f"Few Shot Schema is not in a valid JSON format: {e}") from e

    if query_example:
        try:
            parsed_examples = json.loads(query_example)
            # Validate and retain the description and gremlin fields
            context["query_examples"] = [
                {
                    "description": ex.get("description", ""),
                    "gremlin": ex.get("gremlin", ""),
                }
                for ex in parsed_examples
                if isinstance(ex, dict) and "description" in ex and "gremlin" in ex
            ]
        except json.JSONDecodeError as e:
            raise gr.Error(f"Query Examples is not in a valid JSON format: {e}") from e

    builder = KgBuilder(
        LLMs().get_chat_llm(), Embeddings().get_embedding(), get_hg_client()
    )
    try:
        schema = builder.build_schema().run(context)
    except Exception as e:
        log.error("Failed to generate schema: %s", e)
        raise gr.Error(f"Schema generation failed: {e}") from e
    try:
        formatted_schema = json.dumps(schema, ensure_ascii=False, indent=2)
        return formatted_schema
    except (TypeError, ValueError) as e:
        log.error("Failed to format schema: %s", e)
        return str(schema)
