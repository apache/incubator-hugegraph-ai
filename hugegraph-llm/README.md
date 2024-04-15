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

## Examples

### Examples (knowledge graph construction by llm)

1. Start the HugeGraph database, you can do it via Docker. Refer to this [link](https://hub.docker.com/r/hugegraph/hugegraph) for guidance
2. Run example like `python hugegraph-llm/examples/build_kg_test.py`
 
> Note: If you need a proxy to access OpenAI's API, please set your HTTP proxy in `build_kg_test.py`.

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
