# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import pytest

from hugegraph_ml.utils.dgl2hugegraph_utils import clear_all_data, import_graph_from_dgl, import_graphs_from_dgl


@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown():
    print("Setup: Importing DGL dataset to HugeGraph")
    clear_all_data()
    import_graph_from_dgl("cora")
    import_graphs_from_dgl("MUTAG")

    yield

    print("Teardown: Clearing HugeGraph data")
    clear_all_data()
