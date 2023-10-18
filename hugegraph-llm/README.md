# hugegraph-llm

The `hugegraph-llm` will house the implementation and research related to large language models. It will include runnable demos and can also be used as a third-party library, reducing the cost of using graph systems and the complexity of building knowledge graphs. Graph systems can help large models address challenges like timeliness and hallucination, while large models can assist graph systems with cost-related issues. Therefore, this module will explore more applications and integration solutions for graph systems and large language models.

1.  knowledge graph construction by llm
2.  Use natural language to operate graph databases （gremlin）
3.  Knowledge graph supplements answer context （RAG）

# Examples

## Examples（knowledge graph construction by llm）

1. Start the HugeGraph database, and it is recommended to do so using Docker. Refer to this [link](https://hub.docker.com/r/hugegraph/hugegraph) for guidance
2. Run hugegraph-llm/examples/build_kg_test.py

Note: If you need a proxy to access OpenAI's API, please set your HTTP proxy in `build_kg_test.py`  and 

