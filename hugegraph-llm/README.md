# hugegraph-llm

## Summary

The hugegraph-llm is a tool for the implementation and research related to large language models. This project includes runnable demos, it can also be used as a third-party library.

As we know, graph systems can help large models address challenges like timeliness and hallucination, while large models can assist graph systems with cost-related issues.

With this project, we aim to reduce the cost of using graph systems, and decrease the complexity of building knowledge graphs. This project will offers more applications and integration solutions for graph systems and large language models.
1.  Construct knowledge graph by LLM + HugeGraph
2.  Use natural language to operate graph databases (gremlin)
3.  Knowledge graph supplements answer context (RAG)

# Examples

## Examples（knowledge graph construction by llm）

1. Start the HugeGraph database, you can do it via Docker. Refer to this [link](https://hub.docker.com/r/hugegraph/hugegraph) for guidance
2. Run example like `python hugegraph-llm/examples/build_kg_test.py`

Note: If you need a proxy to access OpenAI's API, please set your HTTP proxy in `build_kg_test.py`.

