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


def clean_vector_index():
    if os.path.exists(os.path.join(resource_path, settings.graph_name, "vidx.faiss")):
        os.remove(os.path.join(resource_path, settings.graph_name, "vidx.faiss"))
    if os.path.exists(os.path.join(resource_path, settings.graph_name, "vidx.pkl")):
        os.remove(os.path.join(resource_path, settings.graph_name, "vidx.pkl"))
    if os.path.exists(os.path.join(resource_path, settings.graph_name, "vid.faiss")):
        os.remove(os.path.join(resource_path, settings.graph_name, "vid.faiss"))
    if os.path.exists(os.path.join(resource_path, settings.graph_name, "vid.pkl")):
        os.remove(os.path.join(resource_path, settings.graph_name, "vid.pkl"))
