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
# under the License.\


import os
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ConfigData:
    """LLM settings"""

    # env_path: Optional[str] = ".env"
    llm_type: Literal["openai", "ollama", "qianfan_wenxin", "zhipu"] = "openai"
    embedding_type: Optional[Literal["openai", "ollama", "qianfan_wenxin", "zhipu"]] = "openai"
    reranker_type: Optional[Literal["cohere", "siliconflow"]] = None
    # 1. OpenAI settings
    openai_api_base: Optional[str] = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
    openai_language_model: Optional[str] = "gpt-4o-mini"
    openai_embedding_model: Optional[str] = "text-embedding-3-small"
    openai_max_tokens: int = 4096
    # 2. Rerank settings
    cohere_base_url: Optional[str] = os.environ.get("CO_API_URL", "https://api.cohere.com/v1/rerank")
    reranker_api_key: Optional[str] = None
    reranker_model: Optional[str] = None
    # 3. Ollama settings
    ollama_host: Optional[str] = "127.0.0.1"
    ollama_port: Optional[int] = 11434
    ollama_language_model: Optional[str] = None
    ollama_embedding_model: Optional[str] = None
    # 4. QianFan/WenXin settings
    qianfan_api_key: Optional[str] = None
    qianfan_secret_key: Optional[str] = None
    qianfan_access_token: Optional[str] = None
    # 4.1 URL settings
    qianfan_url_prefix: Optional[str] = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop"
    qianfan_chat_url: Optional[str] = qianfan_url_prefix + "/chat/"
    qianfan_language_model: Optional[str] = "ERNIE-4.0-Turbo-8K"
    qianfan_embed_url: Optional[str] = qianfan_url_prefix + "/embeddings/"
    # refer https://cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu to get more details
    qianfan_embedding_model: Optional[str] = "embedding-v1"
    # TODO: To be confirmed, whether to configure
    # 5. ZhiPu(GLM) settings
    zhipu_api_key: Optional[str] = None
    zhipu_language_model: Optional[str] = "glm-4"
    zhipu_embedding_model: Optional[str] = "embedding-2"

    """HugeGraph settings"""
    graph_ip: Optional[str] = "127.0.0.1"
    graph_port: Optional[str] = "8080"
    graph_name: Optional[str] = "hugegraph"
    graph_user: Optional[str] = "admin"
    graph_pwd: Optional[str] = "xxx"
    graph_space: Optional[str] = None

    """rag question/answering settings"""
    question: Optional[str] = "Tell me about Sarah."


# Additional static content like PromptConfig
class PromptData:
    rag_schema = """{
"vertexlabels": [
    {
    "id": 1,
    "name": "person",
    "id_strategy": "PRIMARY_KEY",
    "primary_keys": [
        "name"
    ],
    "properties": [
        "name",
        "age",
        "occupation"
    ]
    },
    {
    "id": 2,
    "name": "webpage",
    "id_strategy": "PRIMARY_KEY",
    "primary_keys": [
        "name"
    ],
    "properties": [
        "name",
        "url"
    ]
    }
],
"edgelabels": [
    {
    "id": 1,
    "name": "roommate",
    "source_label": "person",
    "target_label": "person",
    "properties": [
        "date"
    ]
    },
    {
    "id": 2,
    "name": "link",
    "source_label": "webpage",
    "target_label": "person",
    "properties": []
    }
]
}
"""

    # Data is detached from hugegraph-llm/src/hugegraph_llm/operators/llm_op/property_graph_extract.py
    schema_example_prompt = """## Main Task
Given the following graph schema and a piece of text, your task is to analyze the text and extract information that fits into the schema's structure, formatting the information into vertices and edges as specified.

## Basic Rules
### Schema Format
Graph Schema:
- Vertices: [List of vertex labels and their properties]
- Edges: [List of edge labels, their source and target vertex labels, and properties]

### Content Rule
Please read the provided text carefully and identify any information that corresponds to the vertices and edges defined in the schema. For each piece of information that matches a vertex or edge, format it according to the following JSON structures:
#### Vertex Format:
{"id":"vertexLabelID:entityName","label":"vertexLabel","type":"vertex","properties":{"propertyName":"propertyValue", ...}}

#### Edge Format:
{"label":"edgeLabel","type":"edge","outV":"sourceVertexId","outVLabel":"sourceVertexLabel","inV":"targetVertexId","inVLabel":"targetVertexLabel","properties":{"propertyName":"propertyValue",...}}

Also follow the rules: 
1. Don't extract property fields that do not exist in the given schema
2. Ensure the extracted property is in the same type as the schema (like 'age' should be a number)
3. If there are multiple primary keys, the strategy for generating VID is: vertexlabelID:pk1!pk2!pk3 (pk means primary key, and '!' is the separator)
4. Output should be a list of JSON objects, each representing a vertex or an edge, extracted and formatted based on the text and schema.
5. Translate the schema fields into Chinese if the given text is Chinese but the schema is in English (Optional)

## Example
### Input example:
#### text
Meet Sarah, a 30-year-old attorney, and her roommate, James, whom she's shared a home with since 2010. James, in his professional life, works as a journalist.  

#### graph schema
{"vertices":[{"vertex_label":"person","properties":["name","age","occupation"]}], "edges":[{"edge_label":"roommate", "source_vertex_label":"person","target_vertex_label":"person","properties":["date"]]}

### Output example:
[{"id":"1:Sarah","label":"person","type":"vertex","properties":{"name":"Sarah","age":30,"occupation":"attorney"}},{"id":"1:James","label":"person","type":"vertex","properties":{"name":"James","occupation":"journalist"}},{"label":"roommate","type":"edge","outV":"1:Sarah","outVLabel":"person","inV":"1:James","inVLabel":"person","properties":{"date":"2010"}}]
"""

    apache_license_header = """# Licensed to the Apache Software Foundation (ASF) under one
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
"""

