# hugegraph-ai

[![License](https://img.shields.io/badge/license-Apache%202-0E78BA.svg)](https://www.apache.org/licenses/LICENSE-2.0.html)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/apache/incubator-hugegraph-ai)

`hugegraph-ai` aims to explore the integration of [HugeGraph](https://github.com/apache/hugegraph) with artificial 
intelligence (AI) and provide comprehensive support for developers to leverage HugeGraph's AI capabilities 
in their projects.


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
- [vermeer-python-client](./vermeer-python-client): The `vermeer-python-client` is a Python client for Vermeer, a graph data analysis platform. (Note: This module was previously named `vermeer` and its setup has been integrated into the main project).

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for package management and as a workspace manager. All dependencies for the submodules are defined in the root `pyproject.toml`.

### Prerequisites

1.  **Install uv:**
    Follow the official instructions to install `uv` on your system. For example:
    ```bash
    # On macOS and Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # On Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    # Or, if you have pipx or pip globally:
    # pipx install uv
    # pip install uv
    ```

2.  **Create and Activate a Virtual Environment:**
    It's highly recommended to work within a virtual environment. 
    
    If you fail to execute the script using powershell in a Windows environment, [refer to link](https://learn.microsoft.com/zh-cn/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.5)
    
    ```bash
    # Create a virtual environment in .venv directory
    uv venv
    # Activate the environment
    # On macOS and Linux
    source .venv/bin/activate
    # On Windows (PowerShell)  
    .venv\Scripts\Activate.ps1
    ```

### Installing Dependencies

You can install dependencies for all submodules or for specific submodules as needed. The `-e` flag ensures an editable installation, linking directly to your local submodule code.

*   **To install all dependencies (including all submodules):**
    ```bash
    uv pip install -e .[all]
    ```

*   **To install dependencies for a specific submodule:**
    *   For `hugegraph-llm`:
        ```bash
        uv pip install -e .[llm]
        ```
    *   For `hugegraph-ml`:
        ```bash
        uv pip install -e .[ml]
        ```
    *   For `hugegraph-python-client` (core client library):
        ```bash
        uv pip install -e .[python-client]
        ```
    *   For `vermeer-python-client`:
        ```bash
        uv pip install -e .[vermeer]
        ```

Once the dependencies are installed, you can proceed with development across the workspace.

## Learn More

The [project homepage](https://hugegraph.apache.org/docs/quickstart/hugegraph-ai/) contains more information about 
hugegraph-ai.

And here are links of other repositories:
1. [hugegraph](https://github.com/apache/hugegraph) (graph's core component - Graph server + PD + Store)
2. [hugegraph-toolchain](https://github.com/apache/hugegraph-toolchain) (graph tools **[loader](https://github.com/apache/incubator-hugegraph-toolchain/tree/master/hugegraph-loader)/[dashboard](https://github.com/apache/incubator-hugegraph-toolchain/tree/master/hugegraph-hubble)/[tool](https://github.com/apache/incubator-hugegraph-toolchain/tree/master/hugegraph-tools)/[client](https://github.com/apache/incubator-hugegraph-toolchain/tree/master/hugegraph-client)**)
3. [hugegraph-computer](https://github.com/apache/hugegraph-computer) (integrated **graph computing** system)
4. [hugegraph-website](https://github.com/apache/hugegraph-doc) (**doc & website** code)


## Contributing

- Welcome to contribute to HugeGraph, please see [Guidelines](https://hugegraph.apache.org/docs/contribution-guidelines/) for more information.  
- Note: It's recommended to use [GitHub Desktop](https://desktop.github.com/) to greatly simplify the PR and commit process.  
- Code format: Please run [`./style/code_format_and_analysis.sh`](style/code_format_and_analysis.sh) to format your code before submitting a PR. (Use `pylint` to check code style)
- Thank you to all the people who already contributed to HugeGraph!

[![contributors graph](https://contrib.rocks/image?repo=apache/incubator-hugegraph-ai)](https://github.com/apache/incubator-hugegraph-ai/graphs/contributors)

## 📄 License

hugegraph-ai is licensed under [Apache 2.0 License](./LICENSE).

## 📞 Contact Us

- **GitHub Issues**: [Report bugs or request features](https://github.com/apache/incubator-hugegraph-ai/issues) (fastest response)
- **Email**: [dev@hugegraph.apache.org](mailto:dev@hugegraph.apache.org) ([subscription required](https://hugegraph.apache.org/docs/contribution-guidelines/subscribe/))
- **WeChat**: Follow "Apache HugeGraph" official account

<img src="https://raw.githubusercontent.com/apache/hugegraph-doc/master/assets/images/wechat.png" alt="Apache HugeGraph WeChat QR Code" width="200"/>
