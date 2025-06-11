# vermeer-python-client

The `vermeer-python-client` is a Python client(SDK) for [Vermeer](https://github.com/apache/incubator-hugegraph-computer/tree/master/vermeer#readme) (A high-performance distributed graph computing platform based on memory, supporting more than 15 graph algorithms, custom algorithm extensions, and custom data source access & easy to deploy and use)

## Installation

### Install the released package (🚧 ing)

To install the `vermeer-python-client`, you can use uv/pip:

```bash
# uv is optional, you can use pip directly
# uv pip install vermeer-python-client (Not published yet 🚧ing)
# Note: This will install the latest released version. For the newest code, please install from source.
```

### Install from Source (Latest Code)

To install from the source, clone the repository and install the required dependencies:

```bash
git clone https://github.com/apache/incubator-hugegraph-ai.git
cd incubator-hugegraph-ai/vermeer-python-client

# Normal install
uv pip install .

# (Optional) install the devel version
uv pip install -e .
```

## Usage

This section provides examples of how to use the `vermeer-python-client`.

**Note:** The following examples are placeholders. Please replace them with actual usage scenarios for the `vermeer-python-client`.

### Initialize the Client

```python
from pyvermeer.client import VermeerClient

# Initialize the client
client = VermeerClient(host="127.0.0.1", port="8080")
print("Client initialized successfully.")
```

### Example: Running a Graph Algorithm

```python
# Placeholder for running a graph algorithm example
try:
  result = client.run_algorithm(name="pagerank", params={"alpha": 0.85, "max_iter": 10})
  print(f"PageRank results: {result}")
except Exception as e:
  print(f"Error running algorithm: {e}")
```

### Example: Managing Jobs

```python
# Placeholder for managing jobs example
try:
  job_status = client.get_job_status(job_id="some_job_id")
  print(f"Job status: {job_status}")
except Exception as e:
  print(f"Error getting job status: {e}")
```

Other info is under 🚧 (Welcome to add more docs for it)

## Contributing

* Welcome to contribute to `vermeer-python-client`. Please see the [Guidelines](https://hugegraph.apache.org/docs/contribution-guidelines/) for more information.
* Code format: Please run `./style/code_format_and_analysis.sh` to format your code before submitting a PR.

Thank you to all the people who already contributed to `vermeer-python-client`!

## Contact Us

* [GitHub Issues](https://github.com/apache/incubator-hugegraph-ai/issues): Feedback on usage issues and functional requirements (quick response)
* Feedback Email: [dev@hugegraph.apache.org](mailto:dev@hugegraph.apache.org) (subscriber only)
```
