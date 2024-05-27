# hugegraph-llm

## Summary

The `hugegraph-llm` is a tool for the implementation and research related to large language models.
This project includes runnable demos, it can also be used as a third-party library.

As we know, graph systems can help large models address challenges like timeliness and hallucination,
while large models can assist graph systems with cost-related issues.

With this project, we aim to reduce the cost of using graph systems, and decrease the complexity of 
building knowledge graphs. This project will offer more applications and integration solutions for 
graph systems and large language models.
1.  Construct knowledge graph by LLM + HugeGraph
2.  Use natural language to operate graph databases (gremlin)
3.  Knowledge graph supplements answer context (RAG)

## Environment Requirements

- python 3.8+ 
- hugegraph 1.0.0+

## Preparation

- Start the HugeGraph database, you can do it via Docker. Refer to [docker-link](https://hub.docker.com/r/hugegraph/hugegraph) & [deploy-doc](https://hugegraph.apache.org/docs/quickstart/hugegraph-server/#31-use-docker-container-convenient-for-testdev) for guidance
- Start the gradio interactive demo, you can start with the following command, and open http://127.0.0.1:8001 after starting
    ```bash
    # ${PROJECT_ROOT_DIR} is the root directory of hugegraph-ai, which needs to be configured by yourself
    export PYTHONPATH=${PROJECT_ROOT_DIR}/hugegraph-llm/src:${PROJECT_ROOT_DIR}/hugegraph-python-client/src
    python3 ./hugegraph-llm/src/hugegraph_llm/utils/gradio_demo.py
   ```
- Configure HugeGraph database connection information and LLM information, which can be configured in two ways:
  1. Configure the `./hugegraph-llm/src/config/config.ini` file
  2. In gradio, after completing the configurations for LLM and HugeGraph, click on `Initialize configs`, the complete and initialized configuration file will be outputted.
- offline download NLTK stopwords
    ```bash
    python3 ./hugegraph_llm/operators/common_op/nltk_helper.py
    ```

## Examples

### 1.Build a knowledge graph in HugeGraph through LLM

Run example like `python3 ./hugegraph-llm/examples/build_kg_test.py`

The `KgBuilder` class is used to construct a knowledge graph. Here is a brief usage guide:

1. **Initialization**: The `KgBuilder` class is initialized with an instance of a language model. This can be obtained from the `LLMs` class.

    ```python
    from hugegraph_llm.llms.init_llm import LLMs
    from hugegraph_llm.operators.kg_construction_task import KgBuilder
    
    TEXT = ""
    builder = KgBuilder(LLMs().get_llm())
    (
        builder
        .import_schema(from_hugegraph="talent_graph").print_result()
        .extract_triples(TEXT).print_result()
        .disambiguate_word_sense().print_result()
        .commit_to_hugegraph()
        .run()
    )
    ```

2. **Import Schema**: The `import_schema` method is used to import a schema from a source. The source can be a HugeGraph instance, a user-defined schema or an extraction result. The method `print_result` can be chained to print the result.

    ```python
    # Import schema from a HugeGraph instance
    import_schema(from_hugegraph="xxx").print_result()
    # Import schema from an extraction result
    import_schema(from_extraction="xxx").print_result()
    # Import schema from user-defined schema
    import_schema(from_user_defined="xxx").print_result()
    ```

3. **Extract Triples**: The `extract_triples` method is used to extract triples from a text. The text should be passed as a string argument to the method.

    ```python
    TEXT = "Meet Sarah, a 30-year-old attorney, and her roommate, James, whom she's shared a home with since 2010."
    extract_triples(TEXT).print_result()
    ```

4. **Disambiguate Word Sense**: The `disambiguate_word_sense` method is used to disambiguate the sense of words in the extracted triples.

    ```python
    disambiguate_word_sense().print_result()
    ```

5. **Commit to HugeGraph**: The `commit_to_hugegraph` method is used to commit the constructed knowledge graph to a HugeGraph instance.

    ```python
    commit_to_hugegraph().print_result()
    ```

6. **Run**: The `run` method is used to execute the chained operations.

    ```python
    run()
    ```

The methods of the `KgBuilder` class can be chained together to perform a sequence of operations.

### 2. Retrieval augmented generation (RAG) based on HugeGraph

Run example like `python3 ./hugegraph-llm/examples/graph_rag_test.py`

The `GraphRAG` class is used to  integrate HugeGraph with large language models to provide retrieval-augmented generation capabilities. Here is a brief usage guide:

1. **Extract Keyword:**: Extract keywords and expand synonyms.
    
    ```python
    graph_rag.extract_keyword(text="Tell me about Al Pacino.").print_result()
    ```

2. **Query Graph for Rag**: Retrieve the corresponding keywords and their multi-degree associated relationships from HugeGraph.

     ```python
     graph_rag.query_graph_for_rag(
        max_deep=2,
        max_items=30
     ).print_result()
     ```
3. **Synthesize Answer**: Summarize the results and organize the language to answer the question.

    ```python
    graph_rag.synthesize_answer().print_result()
    ```

4. **Run**: The `run` method is used to execute the above operations.

    ```python
    graph_rag.run(verbose=True)
    ```
