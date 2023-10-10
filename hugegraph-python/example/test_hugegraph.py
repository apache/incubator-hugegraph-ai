class HugeGraph:
    """HugeGraph wrapper for graph operations"""

    def __init__(
        self,
        username: str = "default",
        password: str = "default",
        address: str = "127.0.0.1",
        port: int = 8081,
        graph: str = "hugegraph"
    ) -> None:
        """Create a new HugeGraph wrapper instance."""
        try:
            from src.connection import PyHugeClient
        except ImportError:
            raise ValueError(
                "Please install HugeGraph Python client first: "
                "`pip3 install hugegraph-python-client`"
            )

        self.username = username
        self.password = password
        self.address = address
        self.port = port
        self.graph = graph
        self.client = PyHugeClient(address, port, user=username, pwd=password, graph=graph)
        self.schema = ""

    def exec(self, query) -> str:
        """Returns the schema of the HugeGraph database"""
        return self.client.gremlin().exec(query)