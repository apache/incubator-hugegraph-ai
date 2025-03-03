# hugegraph-ai

[![License](https://img.shields.io/badge/license-Apache%202-0E78BA.svg)](https://www.apache.org/licenses/LICENSE-2.0.html)

`hugegraph-ai` aims to explore the integration of HugeGraph with artificial intelligence (AI) and provide comprehensive support for 
developers to leverage HugeGraph's AI capabilities in their projects.


## Modules

- [hugegraph-llm](./hugegraph-llm): The `hugegraph-llm` will house the implementation and research related to large language models.
It will include runnable demos and can also be used as a third-party library, reducing the cost of using graph systems 
and the complexity of building knowledge graphs. Graph systems can help large models address challenges like timeliness 
and hallucination, while large models can help graph systems with cost-related issues. Therefore, this module will 
explore more applications and integration solutions for graph systems and large language models.  (GraphRAG/Agent)
- [hugegraph-ml](./hugegraph-ml): The `hugegraph-ml` will focus on integrating HugeGraph with graph machine learning, 
graph neural networks, and graph embeddings libraries. It will build an efficient and versatile intermediate layer 
to seamlessly connect with third-party graph-related ML frameworks.
- [hugegraph-python-client](./hugegraph-python-client): The `hugegraph-python-client` is a Python client for HugeGraph. 
It is used to define graph structures and perform CRUD operations on graph data. Both the `hugegraph-llm` and 
  `hugegraph-ml` modules will depend on this foundational library. 

## Contributing

- Code format: Please run [`./style/code_format_and_analysis.sh`](style/code_format_and_analysis.sh) to format your code before submitting a PR. (Use `pylint` to check code style)
- Thank you to all the people who already contributed to HugeGraph!

## License

hugegraph-ai is licensed under [Apache 2.0 License](./LICENSE).


## Contact Us

 - 如流 HugeGraph Team/Group
 - GraphRAG DevOps Team (🚧)
