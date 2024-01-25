import unittest

from tests.client_utils import ClientUtils


class TestGraphsManager(unittest.TestCase):
    client = None
    graph = None

    @classmethod
    def setUpClass(cls):
        cls.client = ClientUtils()
        cls.graphs = cls.client.graphs
        cls.client.init_property_key()
        cls.client.init_vertex_label()
        cls.client.init_edge_label()
        cls.client.init_index_label()

    @classmethod
    def tearDownClass(cls):
        cls.client.clear_graph_all_data()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # get_all_graphs
    # get_version
    # get_graph_info
    # clear_graph_all_data
    # get_graph_config

    def test_get_all_graphs(self):
        all_graphs = self.graphs.get_all_graphs()
        self.assertTrue("hugegraph" in all_graphs)

    def test_get_version(self):
        version = self.graphs.get_version()
        self.assertIsNotNone(version)

    def test_get_graph_info(self):
        graph_info = self.graphs.get_graph_info()
        self.assertTrue("backend" in graph_info)

    def test_get_graph_config(self):
        graph_config = self.graphs.get_graph_config()
        self.assertIsNotNone(graph_config)
