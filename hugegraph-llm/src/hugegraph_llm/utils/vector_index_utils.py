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
from typing import Type

import docx
import gradio as gr

from hugegraph_llm.config import huge_settings, index_settings
from hugegraph_llm.flows.scheduler import SchedulerSingleton
from hugegraph_llm.indices.vector_index.base import VectorStoreBase
from hugegraph_llm.indices.vector_index.faiss_vector_store import FaissVectorIndex
from hugegraph_llm.models.embeddings.init_embedding import Embeddings


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
                raise gr.Error(
                    "PDF will be supported later! Try to upload text/docx now"
                )
            else:
                raise gr.Error("Please input txt or docx file.")
    else:
        raise gr.Error("Please input text or upload file.")
    return texts


# pylint: disable=C0301
def get_vector_index_info():
    vector_index = get_vector_index_class(index_settings.cur_vector_index)
    vector_index_entity = vector_index.from_name(
        Embeddings().get_embedding().get_embedding_dim(), huge_settings.graph_name, "chunks"
    )

    return json.dumps(
        {
            **vector_index_entity.get_vector_index_info(),
            "cur_vector_index": index_settings.cur_vector_index,
        },
        ensure_ascii=False,
        indent=2,
    )


def clean_vector_index():
    vector_index = get_vector_index_class(index_settings.cur_vector_index)
    vector_index.clean(huge_settings.graph_name, "chunks")
    gr.Info("Clean vector index successfully!")


def build_vector_index(input_file, input_text):
    if input_file and input_text:
        raise gr.Error("Please only choose one between file and text.")
    texts = read_documents(input_file, input_text)
    scheduler = SchedulerSingleton.get_instance()
    return scheduler.schedule_flow("build_vector_index", texts)


def get_vector_index_class(vector_index_str: str) -> Type[VectorStoreBase]:
    if vector_index_str == "Faiss":
        return FaissVectorIndex  # type: ignore[return-value]
    if vector_index_str == "Milvus":
        try:
            from hugegraph_llm.indices.vector_index.milvus_vector_store import (  # pylint: disable=import-outside-toplevel
                MilvusVectorIndex,
            )

            return MilvusVectorIndex  # type: ignore[return-value]
        except Exception as e:  # pylint: disable=broad-except
            raise gr.Error(
                f"Milvus engine selected but dependency not available: {e}.\n"
                "Fix it by running: 'uv sync --extra vectordb' (recommended) or install 'pymilvus' manually.\n"
                "Alternatively, switch vector engine to Faiss/Qdrant in the UI."
            )
    if vector_index_str == "Qdrant":
        try:
            from hugegraph_llm.indices.vector_index.qdrant_vector_store import (  # pylint: disable=import-outside-toplevel
                QdrantVectorIndex,
            )

            return QdrantVectorIndex  # type: ignore[return-value]
        except Exception as e:  # pylint: disable=broad-except
            raise gr.Error(
                f"Qdrant engine selected but dependency not available: {e}.\n"
                "Fix it by running: 'uv sync --extra vectordb' (recommended) or install 'qdrant-client' manually.\n"
                "Alternatively, switch vector engine to Faiss/Milvus in the UI."
            )
    # Fallback to Faiss
    return FaissVectorIndex  # type: ignore[return-value]
