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

from hugegraph_llm.config import resource_path, settings
from hugegraph_llm.models.embeddings.init_embedding import Embeddings
from hugegraph_llm.operators.index_op.build_semantic_index import BuildSemanticIndex
from hugegraph_llm.utils.hugegraph_utils import get_hg_client
from hugegraph_llm.utils.log import log


def build_index():
    client = get_hg_client()
    context = {"vertices": []}
    vertices = client.gremlin().exec("g.V()")
    for vertex in vertices:
        context["vertices"].append({
            "id": id
        })
    if os.path.exists(os.path.join(resource_path, settings.graph_name, "vid.faiss")):
        log.info("Removing existing semantic id index...")
        os.remove(os.path.join(resource_path, settings.graph_name, "vid.faiss"))
    if os.path.exists(os.path.join(resource_path, settings.graph_name, "vid.pkl")):
        log.info("Removing existing semantic id map...")
        os.remove(os.path.join(resource_path, settings.graph_name, "vid.pkl"))
    build_semantic_index = BuildSemanticIndex(Embeddings().get_embedding())
    context = build_semantic_index.run(context)
    return context


def clean_vector_index():
    if os.path.exists(os.path.join(resource_path, settings.graph_name, "vidx.faiss")):
        os.remove(os.path.join(resource_path, settings.graph_name, "vidx.faiss"))
    if os.path.exists(os.path.join(resource_path, settings.graph_name, "vidx.pkl")):
        os.remove(os.path.join(resource_path, settings.graph_name, "vidx.pkl"))
    if os.path.exists(os.path.join(resource_path, settings.graph_name, "vid.faiss")):
        os.remove(os.path.join(resource_path, settings.graph_name, "vid.faiss"))
    if os.path.exists(os.path.join(resource_path, settings.graph_name, "vid.pkl")):
        os.remove(os.path.join(resource_path, settings.graph_name, "vid.pkl"))
