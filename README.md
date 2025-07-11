# hugegraph-ai

[![License](https://img.shields.io/badge/license-Apache%202-0E78BA.svg)](https://www.apache.org/licenses/LICENSE-2.0.html)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/apache/incubator-hugegraph-ai)

`hugegraph-ai` integrates [HugeGraph](https://github.com/apache/hugegraph) with artificial intelligence capabilities, providing comprehensive support for developers to build AI-powered graph applications.

## ✨ Key Features

- **GraphRAG**: Build intelligent question-answering systems with graph-enhanced retrieval
- **Knowledge Graph Construction**: Automated graph building from text using LLMs
- **Graph ML**: Integration with 20+ graph learning algorithms (GCN, GAT, GraphSAGE, etc.)
- **Python Client**: Easy-to-use Python interface for HugeGraph operations
- **AI Agents**: Intelligent graph analysis and reasoning capabilities

## 🚀 Quick Start

> [!NOTE]
> For a complete deployment guide and detailed examples, please refer to [hugegraph-llm/README.md](./hugegraph-llm/README.md)

### Prerequisites
- Python 3.9+ (3.10+ recommended for hugegraph-llm)
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- HugeGraph Server 1.3+ (1.5+ recommended)
- Docker (optional, for containerized deployment)

### Option 1: Docker Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/apache/incubator-hugegraph-ai.git
cd incubator-hugegraph-ai

# Set up environment and start services
cp docker/env.template docker/.env
# Edit docker/.env to set your PROJECT_PATH
cd docker
# same as `docker-compose` (Legacy)
docker compose -f docker-compose-network.yml up -d

# Access services:
# - HugeGraph Server: http://localhost:8080
# - RAG Service: http://localhost:8001
```

### Option 2: Source Installation

```bash
# 1. Start HugeGraph Server
docker run -itd --name=server -p 8080:8080 hugegraph/hugegraph

# 2. Clone and set up the project
git clone https://github.com/apache/incubator-hugegraph-ai.git
cd incubator-hugegraph-ai/hugegraph-llm

# 3. Install dependencies
uv venv && source .venv/bin/activate
uv pip install -e .

# 4. Start the demo
python -m hugegraph_llm.demo.rag_demo.app
# Visit http://127.0.0.1:8001
```

### Basic Usage Examples

#### GraphRAG - Question Answering
```python
from hugegraph_llm.operators.graph_rag_task import RAGPipeline

# Initialize RAG pipeline
graph_rag = RAGPipeline()

# Ask questions about your graph
result = (graph_rag
    .extract_keywords(text="Tell me about Al Pacino.")
    .keywords_to_vid()
    .query_graphdb(max_deep=2, max_graph_items=30)
    .synthesize_answer()
    .run())
```

#### Knowledge Graph Construction
```python
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.kg_construction_task import KgBuilder

# Build KG from text
TEXT = "Your text content here..."
builder = KgBuilder(LLMs().get_chat_llm())

(builder
    .import_schema(from_hugegraph="hugegraph")
    .chunk_split(TEXT)
    .extract_info(extract_type="property_graph")
    .commit_to_hugegraph()
    .run())
```

#### Graph Machine Learning
```python
from pyhugegraph.client import PyHugeClient
# Connect to HugeGraph and run ML algorithms
# See hugegraph-ml documentation for detailed examples
```

## 📦 Modules

### [hugegraph-llm](./hugegraph-llm) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/apache/incubator-hugegraph-ai)
Large language model integration for graph applications:
- **GraphRAG**: Retrieval-augmented generation with graph data
- **Knowledge Graph Construction**: Build KGs from text automatically  
- **Natural Language Interface**: Query graphs using natural language
- **AI Agents**: Intelligent graph analysis and reasoning

### [hugegraph-ml](./hugegraph-ml)
Graph machine learning with 20+ implemented algorithms:
- **Node Classification**: GCN, GAT, GraphSAGE, APPNP, etc.
- **Graph Classification**: DiffPool, P-GNN, etc.
- **Graph Embedding**: DeepWalk, Node2Vec, GRACE, etc.
- **Link Prediction**: SEAL, GATNE, etc.

### [hugegraph-python-client](./hugegraph-python-client)
Python client for HugeGraph operations:
- **Schema Management**: Define vertex/edge labels and properties
- **CRUD Operations**: Create, read, update, delete graph data
- **Gremlin Queries**: Execute graph traversal queries
- **REST API**: Complete HugeGraph REST API coverage

## 📚 Learn More

- [Project Homepage](https://hugegraph.apache.org/docs/quickstart/hugegraph-ai/)
- [LLM Quick Start Guide](./hugegraph-llm/quick_start.md)
- [DeepWiki AI Documentation](https://deepwiki.com/apache/incubator-hugegraph-ai)

## 🔗 Related Projects

- [hugegraph](https://github.com/apache/hugegraph) - Core graph database
- [hugegraph-toolchain](https://github.com/apache/hugegraph-toolchain) - Development tools (Loader, Dashboard, etc.)
- [hugegraph-computer](https://github.com/apache/hugegraph-computer) - Graph computing system

## 🤝 Contributing

We welcome contributions! Please see our [contribution guidelines](https://hugegraph.apache.org/docs/contribution-guidelines/) for details.

**Development Setup:**
- Use [GitHub Desktop](https://desktop.github.com/) for easier PR management
- Run `./style/code_format_and_analysis.sh` before submitting PRs
- Check existing issues before reporting bugs

[![contributors graph](https://contrib.rocks/image?repo=apache/incubator-hugegraph-ai)](https://github.com/apache/incubator-hugegraph-ai/graphs/contributors)

## 📄 License

hugegraph-ai is licensed under [Apache 2.0 License](./LICENSE).

## 📞 Contact Us

- **GitHub Issues**: [Report bugs or request features](https://github.com/apache/incubator-hugegraph-ai/issues) (fastest response)
- **Email**: [dev@hugegraph.apache.org](mailto:dev@hugegraph.apache.org) ([subscription required](https://hugegraph.apache.org/docs/contribution-guidelines/subscribe/))
- **WeChat**: Follow "Apache HugeGraph" official account

<img src="https://raw.githubusercontent.com/apache/hugegraph-doc/master/assets/images/wechat.png" alt="Apache HugeGraph WeChat QR Code" width="200"/>
