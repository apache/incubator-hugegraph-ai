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

from hugegraph_llm.operators.graph_rag_task import GraphRAG
from hugegraph_llm.utils.gradio_demo import init_hg_test_data

if __name__ == "__main__":
    init_hg_test_data()
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ["OPENAI_API_KEY"] = ""

    graph_rag = GraphRAG()
    result = (
        graph_rag.extract_keyword(text="Tell me about Al Pacino.")
        .print_result()
        .query_graph_for_rag(
            max_deep=2,  # default to 2 if not set
            max_items=30,  # default to 30 if not set
            prop_to_match=None,  # default to None if not set
        )
        .print_result()
        .synthesize_answer()
        .print_result()
        .run(verbose=True)
    )
    print("Query:\n- Tell me about Al Pacino.")
    print(f"Answer:\n- {result['answer']}")
