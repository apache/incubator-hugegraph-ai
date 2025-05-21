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

<<<<<<< HEAD
from hugegraph_llm.config import resource_path, huge_settings, llm_settings
from hugegraph_llm.indices.vector_index.faiss_vector_store import FaissVectorIndex
from hugegraph_llm.models.embeddings.init_embedding import Embeddings, model_map
from hugegraph_llm.flows.scheduler import SchedulerSingleton
from hugegraph_llm.utils.embedding_utils import (
    get_filename_prefix,
    get_index_folder_name,
)
=======
from hugegraph_llm.config import huge_settings, index_settings
from hugegraph_llm.indices.vector_index.base import VectorStoreBase
from hugegraph_llm.indices.vector_index.faiss_vector_store import FaissVectorIndex
from hugegraph_llm.indices.vector_index.milvus_vector_store import MilvusVectorIndex
from hugegraph_llm.indices.vector_index.qdrant_vector_store import QdrantVectorIndex
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
>>>>>>> 38dce0b (feat(llm): vector db finished)
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


# pylint: disable=C0301
def get_vector_index_info():
<<<<<<< HEAD
    folder_name = get_index_folder_name(huge_settings.graph_name, huge_settings.graph_space)
    filename_prefix = get_filename_prefix(
        llm_settings.embedding_type, model_map.get(llm_settings.embedding_type)
    )
    chunk_vector_index = FaissVectorIndex.from_index_file(
        str(os.path.join(resource_path, folder_name, "chunks")),
        filename_prefix,
        record_miss=False,
    )
    graph_vid_vector_index = FaissVectorIndex.from_index_file(
        str(os.path.join(resource_path, folder_name, "graph_vids")), filename_prefix
=======
    vector_index = get_vector_index_class(index_settings.now_vector_index)
    vector_index_entity = vector_index.from_name(
        Embeddings().get_embedding().get_embedding_dim(), huge_settings.graph_name, "chunks"
>>>>>>> 38dce0b (feat(llm): vector db finished)
    )

    return json.dumps(
        vector_index_entity.get_vector_index_info(),
        ensure_ascii=False,
        indent=2,
    )


def clean_vector_index():
<<<<<<< HEAD
    folder_name = get_index_folder_name(huge_settings.graph_name, huge_settings.graph_space)
    filename_prefix = get_filename_prefix(
        llm_settings.embedding_type, model_map.get(llm_settings.embedding_type)
    )
    FaissVectorIndex.clean(str(os.path.join(resource_path, folder_name, "chunks")), filename_prefix)
=======
    vector_index = get_vector_index_class(index_settings.now_vector_index)
    vector_index.clean(huge_settings.graph_name, "chunks")
>>>>>>> 38dce0b (feat(llm): vector db finished)
    gr.Info("Clean vector index successfully!")


def build_vector_index(input_file, input_text):
    vector_index = get_vector_index_class(index_settings.now_vector_index)
    if input_file and input_text:
        raise gr.Error("Please only choose one between file and text.")
    texts = read_documents(input_file, input_text)
<<<<<<< HEAD
    scheduler = SchedulerSingleton.get_instance()
    return scheduler.schedule_flow("build_vector_index", texts)
=======
    builder = KgBuilder(LLMs().get_chat_llm(), Embeddings().get_embedding(), get_hg_client())
    context = builder.chunk_split(texts, "paragraph", "zh").build_vector_index(vector_index).run()
    return json.dumps(context, ensure_ascii=False, indent=2)


def get_vector_index_class(vector_index_str: str) -> Type[VectorStoreBase]:
    mapping = {
        "Faiss": FaissVectorIndex,
        "Milvus": MilvusVectorIndex,
        "Qdrant": QdrantVectorIndex,
    }
    ret = mapping.get(vector_index_str)
    assert ret
    return ret  # type: ignore
>>>>>>> 38dce0b (feat(llm): vector db finished)
