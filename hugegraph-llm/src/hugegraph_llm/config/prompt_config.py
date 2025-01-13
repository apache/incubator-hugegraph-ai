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


from hugegraph_llm.config.models.base_prompt_config import BasePromptConfig


class PromptConfig(BasePromptConfig):
    # Data is detached from llm_op/answer_synthesize.py
    answer_prompt: str = """You are an expert in knowledge graphs and natural language processing.
Your task is to provide a precise and accurate answer based on the given context.

Given the context information and without using fictive knowledge, 
answer the following query in a concise and professional manner.

Context information is below.
---------------------
{context_str}
---------------------
Query: {query_str}
Answer:
"""

    custom_rerank_info: str = """"""

    default_question: str = """Tell me about Sarah."""

    formula_prompt: str = """Please write your answer using Markdown with MathJax syntax, where inline math is wrapped with `$...$`"""

    # Data is detached from hugegraph-llm/src/hugegraph_llm/operators/llm_op/property_graph_extract.py
    extract_graph_prompt: str = """## Main Task
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
1. Don't extract property fields or labels that doesn't exist in the given schema 
2. Ensure the extracted property set in the same type as the given schema (like 'age' should be a number, 'select' should be a boolean)
3. If there are multiple primary keys, the strategy for generating VID is: vertexlabelID:pk1!pk2!pk3 (pk means primary key, and '!' is the separator)
4. Output in JSON format, only include vertexes and edges & remove empty properties, extracted and formatted based on the text/rules and schema
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

    graph_schema: str = """{
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

    # TODO: we should provide a better example to reduce the useless information
    text2gql_graph_schema: str = "hugegraph"

    # Extracted from llm_op/keyword_extract.py
    keywords_extract_prompt: str = """指令：
请对以下文本执行以下任务：
1. 从文本中提取关键词：
  - 最少 0 个，最多 MAX_KEYWORDS 个。
  - 关键词应为具有完整语义的词语或短语，确保信息完整。
2. 识别需改写的关键词：
  - 从提取的关键词中，识别那些在原语境中具有歧义或存在信息缺失的关键词。
3. 生成同义词：
  - 对这些需改写的关键词，生成其在给定语境下的同义词或含义相近的词语。
  - 使用生成的同义词替换原文中的相应关键词。
  - 如果某个关键词没有合适的同义词，则保留该关键词不变。
要求：
- 关键词应为有意义且具体的实体，避免使用无意义或过于宽泛的词语，或单字符的词（例如：“物品”、“动作”、“效果”、“作用”、“的”、“他”）。
- 优先提取主语、动词和宾语，避免提取虚词或助词。
- 保持语义完整性： 抽取的关键词应尽量保持关键词在原语境中语义和信息的完整性（例如：“苹果电脑”应作为一个整体被抽取，而不是被分为“苹果”和“电脑”）。
- 避免泛化： 不要扩展为不相关的泛化类别。
注意：
- 仅考虑语境相关的同义词： 只需考虑给定语境下的关键词的语义近义词和具有类似含义的其他词语。
- 调整关键词长度： 如果关键词相对宽泛，可以根据语境适当增加单个关键词的长度（例如：“违法行为”可以作为一个单独的关键词被抽取，或抽取为“违法”，但不应拆分为“违法”和“行为”）。
输出格式：
- 仅输出一行内容, 以 KEYWORDS: 为前缀，后跟所有关键词或对应的同义词，之间用逗号分隔。抽取的关键词中不允许出现空格或空字符
- 格式示例：
KEYWORDS:关键词1,关键词2,...,关键词n

MAX_KEYWORDS: {max_keywords}
文本：
{question}
"""
    # pylint: disable=C0301
    # keywords_extract_prompt_EN = """
    # Instruction:
    # Please perform the following tasks on the text below:
    # 1. Extract Keywords and Generate Synonyms from the text:
    #   - At least 0, at most {max_keywords} keywords.
    #   - For each keyword, generate its synonyms or possible variant forms.
    # Requirements:
    # - Keywords should be meaningful and specific entities; avoid using meaningless or overly broad terms (e.g., “object,” “the,” “he”).
    # - Prioritize extracting subjects, verbs, and objects; avoid extracting function words or auxiliary words.
    # - Do not expand into unrelated generalized categories.
    # Note:
    # - Only consider semantic synonyms and other words with similar meanings in the given context.
    # Output Format:
    # - Output only one line, prefixed with KEYWORDS:, followed by all keywords and synonyms, separated by commas.No spaces or empty characters are allowed in the extracted keywords.
    # - Format example:
    # KEYWORDS: keyword1, keyword2, ..., keywordN, synonym1, synonym2, ..., synonymN
    # Text:
    # {question}
    # """

    gremlin_generate_prompt = """
You are an expert in graph query language(Gremlin), your role is to understand the schema of the graph and generate 
accurate Gremlin code based on the given instructions.

# Graph Schema:
```json
{schema}
```
# Rule:
1. Could use the vertex ID directly if it's given in the context.
2. The output format must be like:
```gremlin
g.V().limit(10)
```

# Extracted vertex vid:
{vertices}

# Given the example query-gremlin pairs:
{example}

# Generate  gremlin from the following user query.
{query}

The generated gremlin is:
"""

    doc_input_text: str = """Meet Sarah, a 30-year-old attorney, and her roommate, James, whom she's shared a home with since 2010.
James, in his professional life, works as a journalist. Additionally, Sarah is the proud owner of the website 
www.sarahsplace.com, while James manages his own webpage, though the specific URL is not mentioned here. 
These two individuals, Sarah and James, have not only forged a strong personal bond as roommates but have also 
carved out their distinctive digital presence through their respective webpages, showcasing their varied interests 
and experiences.
"""
