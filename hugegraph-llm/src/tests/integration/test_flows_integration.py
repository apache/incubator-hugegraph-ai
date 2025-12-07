import pytest

from hugegraph_llm.demo.rag_demo.rag_block import update_ui_configs
from hugegraph_llm.demo.rag_demo.text2gremlin_block import build_example_vector_index
from hugegraph_llm.flows import FlowName
from hugegraph_llm.flows.scheduler import SchedulerSingleton
from hugegraph_llm.utils.log import log


class TestFlowsIntegration:
    """Flow集成测试 - 验证各个Flow能正常执行不抛异常"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.index_text = """
        梁漱溟年轻时，一日，他与父亲梁济讨论当时一战欧洲的时局，梁济突然问道：“这个世界会好吗？”梁漱溟答：“我相信世界是一天一天往好里去的。”梁济叹道：“能好就好啊！”然后离家，三日后，梁济投湖自尽。晚年梁漱溟回忆自己的一生和跌宕起伏的近代社会，总结了一本书，书名就叫《这个世界会好吗？》。梁漱溟的回答与年轻时一致。但很多人特别是遗老遗少们总在回忆往日的时光，仿佛那是人类的黄金时代。如同鲁迅笔下的九斤老太，整日里念叨着“一代不如一代”。或者极端如梁济，对世界未来充满悲观，一死了之。在今天的时代，很多人认为“世界正变得越来越糟”，这其中不乏知名的知识分子。平克将这种情况称之为「进步恐惧症」，并总结为「认知偏差」。因为每天的新闻报道里总是充斥着战争、恐怖主义、犯罪、污染等坏消息，不是因为这些事情是主流，而是因为它们是热点，导致给人们的印象是世界越来越糟。所谓“好事不出门，坏事传千里”，而在互联网时代，发达的信息传播让坏事传播的更快更广。要纠正这种「可得性偏差」的方法是用数据说话。数字是最能反应趋势，看战争的比例、犯罪死亡人数在总人数的占比，就能看出犯罪是增加了，还是减少了。实际上，从各种数字显示，人类暴力事件在历史呈明显的下降趋势，这在平克之前发表的另一大部头著作《人性中的善良天使：暴力为什么会减少》中详细阐述过。世界变得更好了，说到底就是进步。
        """
        self.scheduler = SchedulerSingleton.get_instance()

    def test_build_knowledge_graph(self):
        try:
            res = self.scheduler.schedule_flow(FlowName.BUILD_VECTOR_INDEX, [self.index_text])
            assert "chunks" in res, "The result of BUILD_VECTOR_INDEX flow should contain 'chunks' field"
            log.info("✓ BUILD_VECTOR_INDEX flow executed successfully")

            schema = """
            {
                "vertexlabels": [
                    {
                    "id": 1,
                    "name": "Person",
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
                    "name": "Book",
                    "id_strategy": "PRIMARY_KEY",
                    "primary_keys": [
                        "title"
                    ],
                    "properties": [
                        "title",
                        "author",
                        "year"
                    ]
                    },
                    {
                    "id": 3,
                    "name": "Concept",
                    "id_strategy": "PRIMARY_KEY",
                    "primary_keys": [
                        "name"
                    ],
                    "properties": [
                        "name",
                        "description"
                    ]
                    }
                ],
                "edgelabels": [
                    {
                    "id": 1,
                    "name": "Wrote",
                    "source_label": "Person",
                    "target_label": "Book",
                    "properties": []
                    },
                    {
                    "id": 2,
                    "name": "Discussed",
                    "source_label": "Person",
                    "target_label": "Concept",
                    "properties": []
                    },
                    {
                    "id": 3,
                    "name": "Believes",
                    "source_label": "Person",
                    "target_label": "Concept",
                    "properties": []
                    }
                ]
            }
            """

            example_prompt = """
            ## Main Task
            Given the following graph schema and a piece of text, your task is to analyze the text and extract information that fits into the schema's structure, formatting the information into vertices and edges as specified.

            ## Basic Rules:
            ### Schema Format:
            Graph Schema:
            - "vertices": [List of vertex labels and their properties]
            - "edges": [List of edge labels, their source and target vertex labels, and properties]

            ### Content Rule:
            Please read the provided text carefully and identify any information that corresponds to the vertices and edges defined in the schema.
            You are not allowed to modify the schema contraints. Your task is to format the provided information into the required schema, without missing any keyword.
            For each piece of information that matches a vertex or edge, format it strictly according to the following JSON structures:

            #### Vertex Format:
            {"id":"vertexLabelID:entityName","label":"vertexLabel","type":"vertex","properties":{"propertyName":"propertyValue", ...}}

            where:
                - "vertexLabelID": int
                - "vertexLabel": str
                - "entityName": str
                - "type": "vertex"
                - "properties": dict

            #### Edge Format:
            {"id":"vertexlabelID:pk1!pk2!pk3", label":"edgeLabel","type":"edge","outV":"sourceVertexId","outVLabel":"sourceVertexLabel","inV":"targetVertexId","inVLabel":"targetVertexLabel","properties":{"propertyName":"propertyValue",...}}

            where:
                - "id": int or str (conditional) (optional)
                - "edgeLabel": str
                - "type": "edge"
                - "outV": str
                - "outVLabel": str
                - "inV": str
                - "inVLabel": str
                - "properties": dict
                - "sourceVertexId": "vertexLabelID:entityName"
                - "targetVertexId": "vertexLabelID:entityName"

            Strictly follow these rules:
            1. Don't extract property fields or labels that doesn't exist in the given schema. Do not generate new information.
            2. Ensure the extracted property set in the same type as the given schema (like 'age' should be a number, 'select' should be a boolean).
            3. If there are multiple primary keys, the strategy for generating VID is: vertexlabelID:pk1!pk2!pk3 (pk means primary key, and '!' is the separator). This id must be generated ONLY if there are multiple primary keys. If there is only one primary key, the strategy for generating VID is: int (sequencially increasing).
            4. Output in JSON format, only include vertexes and edges & remove empty properties, extracted and formatted based on the text/rules and schema.
            5. Translate the schema fields into Chinese if the given text input is Chinese (Optional)

            Refer to the following baseline example to understand the output generation requirements:
            ## Example:
            ### Input example:
            #### text:
            Meet Sarah, a 30-year-old attorney, and her roommate, James, whom she's shared a home with since 2010. James, in his professional life, works as a journalist.

            #### graph schema example:
            {"vertices":[{"vertex_label":"person","properties":["name","age","occupation"]}], "edges":[{"edge_label":"roommate", "source_vertex_label":"person","target_vertex_label":"person","properties":["date"]]}

            ### Output example:
            {"vertices":[{"id":"1:Sarah","label":"person","type":"vertex","properties":{"name":"Sarah","age":30,"occupation":"attorney"}},{"id":"1:James","label":"person","type":"vertex","properties":{"name":"James","occupation":"journalist"}}], "edges":[{"id": 1, "label":"roommate","type":"edge","outV":"1:Sarah","outVLabel":"person","inV":"1:James","inVLabel":"person","properties":{"date":"2010"}}]}
            """
            data = self.scheduler.schedule_flow(
                FlowName.GRAPH_EXTRACT, schema, [self.index_text], example_prompt, "property_graph"
            )
            assert "vertices" in data, "The result of GRAPH_EXTRACT flow should contain 'vertices' field"
            assert "edges" in data, "The result of GRAPH_EXTRACT flow should contain 'edges' field"
            log.info("✓ GRAPH_EXTRACT flow executed successfully")

            res = self.scheduler.schedule_flow(FlowName.IMPORT_GRAPH_DATA, data, schema)
            assert res is not None, "The result of IMPORT_GRAPH_DATA flow should not be None"
            log.info("✓ IMPORT_GRAPH_DATA flow executed successfully")

            self.scheduler.schedule_flow(FlowName.UPDATE_VID_EMBEDDINGS)
            log.info("✓ UPDATE_VID_EMBEDDING flow executed successfully")
        except Exception as e:
            pytest.fail(f"BUILD_VECTOR_INDEX flow failed: {e}")

    def test_schema_generator(self):
        try:
            query_examples = """
            [
            "Property filter: Find all 'person' nodes with age > 30 and return their name and occupation",
            "Relationship traversal: Find all roommates of the person named Alice, and return their name and age",
            "Shortest path: Find the shortest path between Bob and Charlie and show the edge labels along the way",
            "Subgraph match: Find all friend pairs who both follow the same webpage, and return the names and URL",
            "Aggregation: Count the number of people for each occupation and compute their average age",
            "Time-based filter: Find all nodes created after 2025-01-01 and return their name and created_at",
            "Top-N query: List top 10 most visited webpages with their URL and visit_count"
            ]
            """

            few_shot = """
            {
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

            self.scheduler.schedule_flow(FlowName.BUILD_SCHEMA, [self.index_text], query_examples, few_shot)
        except Exception as e:
            pytest.fail(f"BUILD_VECTOR_INDEX flow failed: {e}")

    def test_graph_extract_prompt(self):
        try:
            scenario = "social relationships"
            example_name = "Official Person-Relationship Extraction"

            res = self.scheduler.schedule_flow(FlowName.PROMPT_GENERATE, self.index_text, scenario, example_name)
            assert res is not None, "The result of PROMPT_GENERATE flow should not be None"
        except Exception as e:
            pytest.fail(f"BUILD_VECTOR_INDEX flow failed: {e}")

    def test_rag(self):
        answer_prompt = """
        You are an expert in the fields of knowledge graphs and natural language processing.

        Please provide precise and accurate answers based on the following context information, which is sorted in order of importance from high to low, without using any fabricated knowledge.

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

        keywords_extract_prompt = """
        Instructions:
        Please perform the following tasks on the text below:
        1. Extract, evaluate, and rank keywords from the text:
        - Minimum 0, maximum MAX_KEYWORDS keywords.
        - Keywords should be complete semantic words or phrases, ensuring information completeness, without any changes to the English capitalization.
        - Assign an importance score to each keyword, as a float between 0.0 and 1.0. A higher score indicates a greater contribution to the core idea of the text.
        - Keywords may contain spaces, but must not contain commas or colons.
        - The final list of keywords must be sorted in descending order based on their importance score.
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
        - Output only one line, prefixed with KEYWORDS:, followed by a comma-separated list of items. Each item should be in the format keyword:importance_score(round to two decimal places). If a keyword has been replaced by a synonym, use the synonym as the keyword in the output.
        - Format example:
        KEYWORDS:keyword1:score1,keyword2:score2,...,keywordN:scoreN

        MAX_KEYWORDS: {max_keywords}
        Text:
        {question}

        """

        query = "梁漱溟和梁济的关系是什么？"

        raw_answer = True
        vector_only_answer = False
        graph_only_answer = False
        graph_vector_answer = False
        graph_ratio = 0.6
        rerank_method = "bleu"
        near_neighbor_first = False
        custom_related_information = ""

        graph_search, gremlin_prompt, vector_search = update_ui_configs(
            answer_prompt,
            custom_related_information,
            graph_only_answer,
            graph_vector_answer,
            None,
            keywords_extract_prompt,
            query,
            vector_only_answer,
        )

        res = self.scheduler.schedule_flow(
            FlowName.RAG_RAW,
            query=query,
            vector_search=vector_search,
            graph_search=graph_search,
            raw_answer=raw_answer,
            vector_only_answer=vector_only_answer,
            graph_only_answer=graph_only_answer,
            graph_vector_answer=graph_vector_answer,
            graph_ratio=graph_ratio,
            rerank_method=rerank_method,
            near_neighbor_first=near_neighbor_first,
            custom_related_information=custom_related_information,
            answer_prompt=answer_prompt,
            keywords_extract_prompt=keywords_extract_prompt,
            gremlin_tmpl_num=-1,
            gremlin_prompt=gremlin_prompt,
        )
        assert res is not None, "The result of RAG flow should not be None"

        raw_answer = False
        vector_only_answer = True
        graph_only_answer = False
        graph_vector_answer = False
        res = self.scheduler.schedule_flow(
            FlowName.RAG_VECTOR_ONLY,
            query=query,
            vector_search=vector_search,
            graph_search=graph_search,
            raw_answer=raw_answer,
            vector_only_answer=vector_only_answer,
            graph_only_answer=graph_only_answer,
            graph_vector_answer=graph_vector_answer,
            graph_ratio=graph_ratio,
            rerank_method=rerank_method,
            near_neighbor_first=near_neighbor_first,
            custom_related_information=custom_related_information,
            answer_prompt=answer_prompt,
            keywords_extract_prompt=keywords_extract_prompt,
            gremlin_tmpl_num=-1,
            gremlin_prompt=gremlin_prompt,
        )
        assert res is not None, "The result of RAG flow should not be None"

        raw_answer = False
        vector_only_answer = False
        graph_only_answer = True
        graph_vector_answer = False
        res = self.scheduler.schedule_flow(
            FlowName.RAG_GRAPH_ONLY,
            query=query,
            vector_search=vector_search,
            graph_search=graph_search,
            raw_answer=raw_answer,
            vector_only_answer=vector_only_answer,
            graph_only_answer=graph_only_answer,
            graph_vector_answer=graph_vector_answer,
            graph_ratio=graph_ratio,
            rerank_method=rerank_method,
            near_neighbor_first=near_neighbor_first,
            custom_related_information=custom_related_information,
            answer_prompt=answer_prompt,
            keywords_extract_prompt=keywords_extract_prompt,
            gremlin_tmpl_num=-1,
            gremlin_prompt=gremlin_prompt,
        )
        assert res is not None, "The result of RAG flow should not be None"

        raw_answer = False
        vector_only_answer = False
        graph_only_answer = False
        graph_vector_answer = True
        res = self.scheduler.schedule_flow(
            FlowName.RAG_GRAPH_VECTOR,
            query=query,
            vector_search=vector_search,
            graph_search=graph_search,
            raw_answer=raw_answer,
            vector_only_answer=vector_only_answer,
            graph_only_answer=graph_only_answer,
            graph_vector_answer=graph_vector_answer,
            graph_ratio=graph_ratio,
            rerank_method=rerank_method,
            near_neighbor_first=near_neighbor_first,
            custom_related_information=custom_related_information,
            answer_prompt=answer_prompt,
            keywords_extract_prompt=keywords_extract_prompt,
            gremlin_tmpl_num=-1,
            gremlin_prompt=gremlin_prompt,
        )
        assert res is not None, "The result of RAG flow should not be None"

    def test_build_example_index(self):
        res = build_example_vector_index(None)
        assert "embed_dim" in res, "The result of build_example_vector_index should contain embed_dim"

    def test_text_2_gremlin(self):
        query = "梁漱溟和梁济的关系是什么？"
        schema = "hugegraph"
        example_num = 2
        gremlin_prompt_input = """

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
        - Similarly, if the user's query refers to specific property names or their values, and these are present or align with the 'Referenced Extracted Properties', actively utilize these properties in your Gremlin query.
        For instance, you can use them for filtering vertices or edges (e.g., using `has('propertyName', 'propertyValue')`), or for projecting specific values.

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

        Referenced Extracted Properties Related to the Query (Format: [('property_name', 'property_value'), ...]):
        {properties}

        Generate Gremlin from the Following User Query:
        {query}

        **Important: Do NOT output any analysis, reasoning steps, explanations or any other text. ONLY return the Gremlin query wrapped in a code block with ```gremlin``` fences.**

        The generated Gremlin is:

        """

        res = self.scheduler.schedule_flow(
            FlowName.TEXT2GREMLIN, query, example_num, schema, gremlin_prompt_input, None
        )

        assert res is not None, "The result of TEXT2GREMLIN flow should not be None"
