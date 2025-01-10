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

from typing import Optional
from .models import BaseConfig


class HugeGraphConfig(BaseConfig):
    """HugeGraph settings"""
    # graph server config
    graph_ip: Optional[str] = "127.0.0.1"
    graph_port: Optional[str] = "8080"
    graph_name: Optional[str] = "hugegraph"
    graph_user: Optional[str] = "admin"
    graph_pwd: Optional[str] = "xxx"
    graph_space: Optional[str] = None

    # graph query config
    limit_property: Optional[str] = "False"
    max_graph_path: Optional[int] = 10
    max_graph_items: Optional[int] = 30
    edge_limit_pre_label: Optional[int] = 8

    # vector config
    vector_dis_threshold: Optional[float] = 0.9
    topk_per_keyword: Optional[int] = 1

    # rerank config
    topk_return_results: Optional[int] = 20
