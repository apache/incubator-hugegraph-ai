import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from example.test_hugegraph import HugeGraph

class HugeGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.username = "test_user"
        self.password = "test_password"
        self.address = "test_address"
        self.graph = "test_hugegraph"
        self.port = 1234
        self.session_pool_size = 10

    @patch("src.client.PyHugeGraph")
    def test_init(self, a: Any) -> None:
        a.return_value = MagicMock()
        client = HugeGraph(self.username, self.password,self.address, self.port, self.graph)

        result = client.exec("g.V().limit(10)")
        self.assertIsInstance(result, MagicMock)

