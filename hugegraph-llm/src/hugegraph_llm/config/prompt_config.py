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

# pylint: disable=C0301
class PromptConfig(BasePromptConfig):
    # Data is detached from llm_op/answer_synthesize.py
    answer_prompt: str = """You are an expert in knowledge graphs and natural language processing.
Your task is to provide a precise and accurate answer based on the given context.

Given the context information and without using fictive knowledge, 
answer the following query in a concise and professional manner.
Please write your answer using Markdown with MathJax syntax, where inline math is wrapped with `$...$`

Context information is below.
---------------------
{context_str}
---------------------
Query: {query_str}
Answer:
"""

    custom_rerank_info: str = """"""

    default_question: str = """Tell me about Sarah."""

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
    keywords_extract_prompt: str = """Instructions:
    Please perform the following tasks on the text below:
    1. Extract keywords from the text:
       - Minimum 0, maximum MAX_KEYWORDS keywords.
       - Keywords should be complete semantic words or phrases, ensuring information completeness.
    2. Identify keywords that need rewriting:
       - From the extracted keywords, identify those that are ambiguous or lack information in the original context.
    3. Generate synonyms:
       - For these keywords that need rewriting, generate synonyms or similar terms in the given context.
       - Replace the corresponding keywords in the original text with generated synonyms.
       - If no suitable synonym exists for a keyword, keep the original keyword unchanged.
    
    Requirements:
    - Keywords should be meaningful and specific entities; avoid meaningless or overly broad terms, or single-character words (e.g., "items", "actions", "effects", "functions", "the", "he").
    - Prioritize extracting subjects, verbs, and objects; avoid function words or auxiliary words.
    - Maintain semantic integrity: Extracted keywords should preserve their semantic and informational completeness in the original context (e.g., "Apple computer" should be extracted as a whole, not split into "Apple" and "computer").
    - Avoid generalization: Do not expand into unrelated generalized categories.
    
    Notes:
    - Only consider context-relevant synonyms: Only consider semantic synonyms and words with similar meanings in the given context.
    - Adjust keyword length: If keywords are relatively broad, you can appropriately increase individual keyword length based on context (e.g., "illegal behavior" can be extracted as a single keyword, or as "illegal", but should not be split into "illegal" and "behavior").
    
    Output Format:
    - Output only one line, prefixed with KEYWORDS:, followed by all keywords or corresponding synonyms, separated by commas. No spaces or empty characters are allowed in the extracted keywords.
    - Format example:
    KEYWORDS:keyword1,keyword2,...,keywordN
    
    MAX_KEYWORDS: {max_keywords}
    Text:
    {question}
    """

    gremlin_generate_prompt = """
You are an expert in graph query language (Gremlin). Your role is to understand the schema of the graph, recognize the intent behind user queries, and generate accurate Gremlin code based on the given instructions.

### Tasks
## Complex Query Detection:
Assess the user's query to determine its complexity based on the following criteria:

1. Multiple Reasoning Steps: The query requires several logical steps to arrive at the final result.
2. Conditional Logic: The query includes multiple conditions or filters that depend on each other.
3. Nested Queries: The query contains sub-queries or nested logical statements.
4. High-Level Abstractions: The query requests high-level summaries or insights that require intricate data manipulation.

# Examples of Complex Queries:
“Retrieve all users who have posted more than five articles and have at least two comments with a positive sentiment score.”
“Calculate the average response time of servers in each data center and identify which data centers are below the required performance threshold after the latest update.”

# Rules
- **Complex Query Handling**:
    - **Detection**: If the user's query meets **any** of the complexity criteria listed above, it is considered **complex**.
    - **Response**: For complex queries, **do not** proceed to Gremlin Query Generation. Instead, directly return the following Gremlin query:
    ```gremlin
    g.V().limit(0)
    ```
- **Simple Query Handling**:
    - If the query does **not** meet any of the complexity criteria, it is considered **simple**.
    - Proceed to the Gremlin Query Generation task as outlined below.

## Gremlin Query Generation (Executed only if the query is not complex):
# Rules
- You may use the vertex ID directly if it’s provided in the context.
- If the provided question contains entity names that are very similar to the Vertices IDs, then in the generated Gremlin statement, replace the approximate entities from the original question.
For example, if the question includes the name ABC, and the provided VerticesIDs do not contain ABC but only abC, then use abC instead of ABC from the original question when generating the gremlin.

The output format must be as follows:
```gremlin
g.V().limit(10)
```
Graph Schema:
{schema}
Refer Gremlin Example Pair:
{example}

Referenced Extracted Vertex IDs Related to the Query:
{vertices}

Generate Gremlin from the Following User Query:
{query}
The generated Gremlin is:
"""

    doc_input_text: str = """Meet Sarah, a 30-year-old attorney, and her roommate, James, whom she's shared a home with since 2010.
James, in his professional life, works as a journalist. Additionally, Sarah is the proud owner of the website
www.sarahsplace.com, while James manages his own webpage, though the specific URL is not mentioned here.
These two individuals, Sarah and James, have not only forged a strong personal bond as roommates but have also
carved out their distinctive digital presence through their respective webpages, showcasing their varied interests
and experiences.
"""

    # TODO: we should switch the prompt automatically based on the language (like using context['language'])
    answer_prompt_CN: str = """你是知识图谱和自然语言处理领域的专家。
你的任务是基于给定的上下文提供精确和准确的答案。

根据提供的上下文信息，不使用虚构知识，
请以简洁专业的方式回答以下问题。
请使用 Markdown 格式编写答案，其中行内数学公式用 `$...$` 包裹

上下文信息如下：
---------------------
{context_str}
---------------------
问题：{query_str}
答案：
"""

    extract_graph_prompt_CN: str = """## 主要任务
根据以下图谱和一段文本，你的任务是分析文本并提取符合模式结构的信息，将信息格式化为顶点和边。

## 基本规则
### 模式格式
图谱模式：
- 顶点：[顶点标签及其属性列表]
- 边：[边标签、源顶点标签、目标顶点标签及其属性列表]

### 内容规则
请仔细阅读提供的文本，识别与模式中定义的顶点和边相对应的信息。对于每一条匹配顶点或边的信息，按以下JSON结构格式化：

#### 顶点格式：
{"id":"顶点标签ID:实体名称","label":"顶点标签","type":"vertex","properties":{"属性名":"属性值", ...}}

#### 边格式：
{"label":"边标签","type":"edge","outV":"源顶点ID","outVLabel":"源顶点标签","inV":"目标顶点ID","inVLabel":"目标顶点标签","properties":{"属性名":"属性值",...}}

同时遵循以下规则：
1. 不要提取给定模式中不存在的属性字段或标签
2. 确保提取的属性集与给定模式类型一致（如'age'应为数字，'select'应为布尔值）
3. 如果有多个主键，生成VID的策略是：顶点标签ID:pk1!pk2!pk3（pk表示主键，'!'是分隔符）
4. 以JSON格式输出，仅包含顶点和边，移除空属性，基于文本/规则和模式提取和格式化
5. 如果给定文本为中文但模式为英文，则将模式字段翻译成中文（可选）

## 示例
### 输入示例：
#### 文本
认识Sarah，一位30岁的律师，和她的室友James，他们从2010年开始合住。James在职业生活中是一名记者。

#### 图谱模式
{"vertices":[{"vertex_label":"person","properties":["name","age","occupation"]}], "edges":[{"edge_label":"roommate", "source_vertex_label":"person","target_vertex_label":"person","properties":["date"]]}

### 输出示例：
[{"id":"1:Sarah","label":"person","type":"vertex","properties":{"name":"Sarah","age":30,"occupation":"律师"}},{"id":"1:James","label":"person","type":"vertex","properties":{"name":"James","occupation":"记者"}},{"label":"roommate","type":"edge","outV":"1:Sarah","outVLabel":"person","inV":"1:James","inVLabel":"person","properties":{"date":"2010"}}]
"""

    gremlin_generate_prompt_CN: str = """
你是图查询语言（Gremlin）的专家。你的角色是理解图谱的模式，识别用户查询背后的意图，并根据给定的指令生成准确的Gremlin代码。

### 任务
## 复杂查询检测：
根据以下标准评估用户的查询以确定其复杂性：

1. 多步推理：查询需要多个逻辑步骤才能得出最终结果。
2. 条件逻辑：查询包含多个相互依赖的条件或过滤器。
3. 嵌套查询：查询包含子查询或嵌套逻辑语句。
4. 高层次抽象：查询请求需要复杂数据操作的高层次总结或见解。

# 复杂查询示例：
"检索发表超过五篇文章且至少有两条积极情感评分评论的所有用户。"
"计算每个数据中心服务器的平均响应时间，并识别最新更新后性能低于要求阈值的数据中心。"

# 规则
- **复杂查询处理**：
    - **检测**：如果用户的查询符合上述任一复杂性标准，则视为**复杂**查询。
    - **响应**：对于复杂查询，**不要**进行Gremlin查询生成。相反，直接返回以下Gremlin查询：
    ```gremlin
    g.V().limit(0)
    ```
- **简单查询处理**：
    - 如果查询**不**符合任何复杂性标准，则视为**简单**查询。
    - 按照下面的说明进行Gremlin查询生成任务。

## Gremlin查询生成（仅在查询不复杂时执行）：
# 规则
- 如果在上下文中提供了顶点ID，可以直接使用。
- 如果提供的问题包含与顶点ID非常相似的实体名称，则在生成的Gremlin语句中替换原始问题中的近似实体。
例如，如果问题包含名称ABC，而提供的顶点ID不包含ABC而只有abC，则在生成gremlin时使用abC而不是原始问题中的ABC。

输出格式必须如下：
```gremlin
g.V().limit(10)
```
图谱模式：
{schema}
参考Gremlin示例对：
{example}

与查询相关的已提取顶点ID：
{vertices}

从以下用户查询生成Gremlin：
{query}
生成的Gremlin是：
"""

    keywords_extract_prompt_CN: str = """指令：
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

    doc_input_text_CN: str = """介绍一下Sarah，她是一位30岁的律师，还有她的室友James，他们从2010年开始一起合租。James是一名记者，
职业道路也很出色。另外，Sarah拥有一个个人网站www.sarahsplace.com，而James也经营着自己的网页，不过这里没有提到具体的网址。这两个人，
Sarah和James，不仅建立起了深厚的室友情谊，还各自在网络上开辟了自己的一片天地，展示着他们各自丰富多彩的兴趣和经历。
"""
