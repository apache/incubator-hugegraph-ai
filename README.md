# hugegraph-ai

[![License](https://img.shields.io/badge/license-Apache%202-0E78BA.svg)](https://www.apache.org/licenses/LICENSE-2.0.html)

`hugegraph-ai` aims to explore the integration of [HugeGraph](https://github.com/apache/hugegraph) with artificial 
intelligence (AI) and provide comprehensive support for developers to leverage HugeGraph's AI capabilities 
in their projects.


## Modules

- [hugegraph-llm](./hugegraph-llm):The `hugegraph-llm` will house the implementation and research related to large language models.
It will include runnable demos and can also be used as a third-party library, reducing the cost of using graph systems 
and the complexity of building knowledge graphs. Graph systems can help large models address challenges like timeliness 
and hallucination, while large models can assist graph systems with cost-related issues. Therefore, this module will 
explore more applications and integration solutions for graph systems and large language models. 
- [hugegraph-ml](./hugegraph-ml): The `hugegraph-ml` will focus on integrating HugeGraph with graph machine learning, 
graph neural networks, and graph embeddings libraries. It will build an efficient and versatile intermediate layer 
to seamlessly connect with third-party graph-related ML frameworks.
- [hugegraph-python-client](./hugegraph-python-client): The `hugegraph-python-client` is a Python client for HugeGraph. 
It is used to define graph structures and perform CRUD operations on graph data. Both the `hugegraph-llm` and `hugegraph-ml` 
modules will depend on this foundational library. 


## Contributing

- Welcome to contribute to HugeGraph, please see [Guidelines](https://hugegraph.apache.org/docs/contribution-guidelines/) for more information.  
- Note: It's recommended to use [GitHub Desktop](https://desktop.github.com/) to greatly simplify the PR and commit process.  
- Code format: Please run [`./style/code_format_and_analysis.sh`](style/code_format_and_analysis.sh) to format your code before submitting a PR.
- Thank you to all the people who already contributed to HugeGraph!

[![contributors graph](https://contrib.rocks/image?repo=apache/incubator-hugegraph-ai)](https://github.com/apache/incubator-hugegraph-ai/graphs/contributors)


## License

hugegraph-ai is licensed under [Apache 2.0 License](./LICENSE).


## Contact Us

 - [GitHub Issues](https://github.com/apache/incubator-hugegraph-ai/issues): Feedback on usage issues and functional requirements (quick response)
 - Feedback Email: [dev@hugegraph.apache.org](mailto:dev@hugegraph.apache.org) ([subscriber](https://hugegraph.apache.org/docs/contribution-guidelines/subscribe/) only)
 - WeChat public account: Apache HugeGraph, welcome to scan this QR code to follow us.

 <img src="https://raw.githubusercontent.com/apache/incubator-hugegraph-doc/master/assets/images/wechat.png" alt="QR png" width="350"/>
