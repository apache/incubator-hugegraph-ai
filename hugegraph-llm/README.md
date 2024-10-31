# hugegraph-llm

## Summary

The `hugegraph-llm` is a tool for the implementation and research related to large language models.
This project includes runnable demos, it can also be used as a third-party library.

As we know, graph systems can help large models address challenges like timeliness and hallucination,
while large models can help graph systems with cost-related issues.

With this project, we aim to reduce the cost of using graph systems, and decrease the complexity of 
building knowledge graphs. This project will offer more applications and integration solutions for 
graph systems and large language models.
1.  Construct knowledge graph by LLM + HugeGraph
2.  Use natural language to operate graph databases (Gremlin/Cypher)
3.  Knowledge graph supplements answer context (GraphRAG)

## Environment Requirements

- python 3.9+ 
- hugegraph-server 1.2+

## Preparation

1. Start the HugeGraph database, you can run it via [Docker](https://hub.docker.com/r/hugegraph/hugegraph)/[Binary Package](https://hugegraph.apache.org/docs/download/download/).
Refer to detailed [doc](https://hugegraph.apache.org/docs/quickstart/hugegraph-server/#31-use-docker-container-convenient-for-testdev) for more guidance (PS: Graph visualization in step8)
2. Clone this project
    ```bash
    git clone https://github.com/apache/incubator-hugegraph-ai.git
    ```
3. Install [hugegraph-python-client](../hugegraph-python-client) and [hugegraph_llm](src/hugegraph_llm)
    ```bash
    cd ./incubator-hugegraph-ai # better to use virtualenv (source venv/bin/activate) 
    pip install ./hugegraph-python-client
    pip install -r ./hugegraph-llm/requirements.txt
    ```
4. Enter the project directory
    ```bash
    cd ./hugegraph-llm/src
    ```

5. Start the gradio interactive demo of **Graph RAG**, you can run with the following command, and open http://127.0.0.1:8001 after starting
    ```bash
    python3 -m hugegraph_llm.demo.rag_demo.app
    ```
    The default host is `0.0.0.0` and the port is `8001`. You can change them by passing command line arguments`--host` and `--port`.  
    ```bash
    python3 -m hugegraph_llm.demo.rag_demo.app --host 127.0.0.1 --port 18001
    ```

6. Or start the gradio interactive demo of **Text2Gremlin**, you can run with the following command, and open http://127.0.0.1:8002 after starting. You can also change the default host `0.0.0.0` and port `8002` as above. (🚧ing)
    ```bash
    python3 -m hugegraph_llm.demo.gremlin_generate_web_demo
   ```

7. After running the web demo, the config file `.env` will be automatically generated. You can modify its content on the web page. Or modify the file directly and restart the web application.

    (Optional)To regenerate the config file, you can use `config.generate` with `-u` or `--update`.
    ```bash
    python3 -m hugegraph_llm.config.generate --update
    ```
   
8. (__Optional__) You could use 
[hugegraph-hubble](https://hugegraph.apache.org/docs/quickstart/hugegraph-hubble/#21-use-docker-convenient-for-testdev) 
to visit the graph data, could run it via [Docker/Docker-Compose](https://hub.docker.com/r/hugegraph/hubble) 
for guidance. (Hubble is a graph-analysis dashboard include data loading/schema management/graph traverser/display).


## Examples

### 1.Build a knowledge graph in HugeGraph through LLM

The `KgBuilder` class is used to construct a knowledge graph. Here is a brief usage guide:

1. **Initialization**: The `KgBuilder` class is initialized with an instance of a language model. 
This can be obtained from the `LLMs` class.

    ```python
    from hugegraph_llm.models.llms.init_llm import LLMs
    from hugegraph_llm.operators.kg_construction_task import KgBuilder
    
    TEXT = ""
    builder = KgBuilder(LLMs().get_llm())
    (
        builder
        .import_schema(from_hugegraph="talent_graph").print_result()
        .chunk_split(TEXT).print_result()
        .extract_info(extract_type="property_graph").print_result()
        .commit_to_hugegraph()
        .run()
    )
    ```

2. **Import Schema**: The `import_schema` method is used to import a schema from a source. The source can be a HugeGraph instance, a user-defined schema or an extraction result. The method `print_result` can be chained to print the result.

    ```python
    # Import schema from a HugeGraph instance
    builder.import_schema(from_hugegraph="xxx").print_result()
    # Import schema from an extraction result
    builder.import_schema(from_extraction="xxx").print_result()
    # Import schema from user-defined schema
    builder.import_schema(from_user_defined="xxx").print_result()
    ```

3. **Chunk Split**: The `chunk_split` method is used to split the input text into chunks. The text should be passed as a string argument to the method.

    ```python
    # Split the input text into documents
    builder.chunk_split(TEXT, split_type="document").print_result()
    # Split the input text into paragraphs
    builder.chunk_split(TEXT, split_type="paragraph").print_result()
    # Split the input text into sentences
    builder.chunk_split(TEXT, split_type="sentence").print_result()
    ```

4. **Extract Info**: The `extract_info` method is used to extract info from a text. The text should be passed as a string argument to the method.

    ```python
    TEXT = "Meet Sarah, a 30-year-old attorney, and her roommate, James, whom she's shared a home with since 2010."
    # extract property graph from the input text
    builder.extract_info(extract_type="property_graph").print_result()
    # extract triples from the input text
    builder.extract_info(extract_type="property_graph").print_result()
    ```

5. **Commit to HugeGraph**: The `commit_to_hugegraph` method is used to commit the constructed knowledge graph to a HugeGraph instance.

    ```python
    builder.commit_to_hugegraph().print_result()
    ```

6. **Run**: The `run` method is used to execute the chained operations.

    ```python
    builder.run()
    ```

The methods of the `KgBuilder` class can be chained together to perform a sequence of operations.

### 2. Retrieval augmented generation (RAG) based on HugeGraph

Run example like `python3 ./hugegraph_llm/examples/graph_rag_test.py`

The `RAGPipeline` class is used to integrate HugeGraph with large language models to provide retrieval-augmented generation capabilities.
Here is a brief usage guide:

1. **Extract Keyword**: Extract keywords and expand synonyms.

    ```python
    from hugegraph_llm.operators.graph_rag_task import RAGPipeline
    graph_rag = RAGPipeline()
    graph_rag.extract_keywords(text="Tell me about Al Pacino.").print_result()
    ```

2. **Match Vid from Keywords*: Match the nodes with the keywords in the graph.

    ```python
    graph_rag.keywords_to_vid().print_result()
    ```

3. **Query Graph for Rag**: Retrieve the corresponding keywords and their multi-degree associated relationships from HugeGraph.

     ```python
     graph_rag.query_graphdb(max_deep=2, max_items=30).print_result()
     ```

4. **Rerank Searched Result**: Rerank the searched results based on the similarity between the question and the results.

     ```python
     graph_rag.merge_dedup_rerank().print_result()
     ```

5. **Synthesize Answer**: Summarize the results and organize the language to answer the question.

    ```python
    graph_rag.synthesize_answer(vector_only_answer=False, graph_only_answer=True).print_result()
    ```

6. **Run**: The `run` method is used to execute the above operations.

    ```python
    graph_rag.run(verbose=True)
    ```
