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

"""Check inline edge sort keys through HTTP, schema preparation and review."""

import json
from copy import deepcopy

import pytest

from tests.integration import test_graph_extract_review as review_tests
from tests.integration.test_graph_extract_review import PASS, SCHEMA, graph, install_llm, post

client = review_tests.client

SORT_SCHEMA = deepcopy(SCHEMA)
SORT_SCHEMA["edgelabels"] = [
    {
        "name": "knows",
        "source_label": "person",
        "target_label": "person",
        "properties": ["at", "note"],
        "frequency": "MULTIPLE",
        "sort_keys": ["at"],
    }
]
SORT_SCHEMA["propertykeys"] += [
    {"name": "at", "data_type": "LONG", "cardinality": "SINGLE"},
    {"name": "note", "data_type": "TEXT", "cardinality": "SINGLE"},
]
TOKEN = "integration_token_edge_sort_key"


def edge_graph(properties):
    return {
        "vertices": graph("Alice")["vertices"] + graph("Bob")["vertices"],
        "edges": [
            {
                "label": "knows",
                "outV": "1:Alice",
                "outVLabel": "person",
                "inV": "1:Bob",
                "inVLabel": "person",
                "properties": properties,
            }
        ],
    }


@pytest.mark.parametrize("explicit_nullable", [False, True])
def test_missing_sort_key_is_blocked_before_review(client, monkeypatch, explicit_nullable):
    schema = deepcopy(SORT_SCHEMA)
    if explicit_nullable:
        schema["edgelabels"][0]["nullable_keys"] = ["at", "note"]
    clients = install_llm(monkeypatch, {TOKEN: [edge_graph({}), PASS]})

    response = post(client, TOKEN, schema=schema, review_options={"max_fixes": 0})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert data["result"] == {"vertices": [], "edges": []}
    chunk = data["chunk_results"][0]
    assert chunk["status"] == "blocked"
    assert chunk["reason"] == "validation_failed"
    assert (chunk["reviews_used"], chunk["fixes_used"], chunk["llm_calls"]) == (0, 0, 1)
    assert [issue["path"] for issue in chunk["issues"]] == ["edges[0].properties.at"]
    assert chunk["graph"]["edges"][0]["properties"] == {}
    assert len(clients.clients[0].prompts) == 1


@pytest.mark.parametrize("repaired", [False, True])
def test_sort_key_repair_is_revalidated_before_review(client, monkeypatch, repaired):
    properties = {"at": 1} if repaired else {}
    clients = install_llm(monkeypatch, {TOKEN: [edge_graph({}), edge_graph(properties), PASS]})

    response = post(client, TOKEN, schema=SORT_SCHEMA, review_options={"max_fixes": 1})

    assert response.status_code == 200
    data = response.json()
    chunk = data["chunk_results"][0]
    repair_context = json.loads(clients.clients[0].prompts[1].split("\n## Input\n")[1])
    assert repair_context["schema"]["edgelabels"][0]["sort_keys"] == ["at"]
    assert "nullable_keys" not in repair_context["schema"]["edgelabels"][0]
    assert [issue["path"] for issue in repair_context["issues"]] == ["edges[0].properties.at"]
    assert repair_context["graph"]["edges"][0]["properties"] == {}
    assert chunk["graph"]["edges"][0]["properties"] == properties
    if repaired:
        assert data["status"] == "succeeded"
        assert chunk["status"] == "final"
        assert chunk["issues"] == []
        assert (chunk["reviews_used"], chunk["fixes_used"], chunk["llm_calls"]) == (1, 1, 3)
        assert data["result"]["edges"][0]["properties"] == {"at": 1}
        review_context = json.loads(clients.clients[0].prompts[2].split("\n## Input\n")[1])
        assert review_context["graph"]["edges"][0]["properties"] == {"at": 1}
    else:
        assert data["status"] == "failed"
        assert chunk["status"] == "blocked"
        assert chunk["reason"] == "validation_failed"
        assert (chunk["reviews_used"], chunk["fixes_used"], chunk["llm_calls"]) == (0, 1, 2)
        assert [issue["path"] for issue in chunk["issues"]] == ["edges[0].properties.at"]
        assert data["result"] == {"vertices": [], "edges": []}
    assert len(clients.clients[0].prompts) == chunk["llm_calls"]


@pytest.mark.parametrize(
    ("sort_keys", "properties", "explicit_nullable"),
    [
        pytest.param(["at"], {"at": 1}, False, id="valid-sort-key-optional-note-absent"),
        pytest.param(["at"], {"at": 1}, True, id="sort-key-listed-as-nullable"),
        pytest.param([], {}, False, id="ordinary-properties-remain-optional"),
    ],
)
def test_sort_keys_do_not_make_ordinary_properties_required(
    client, monkeypatch, sort_keys, properties, explicit_nullable
):
    schema = deepcopy(SORT_SCHEMA)
    schema["edgelabels"][0]["sort_keys"] = sort_keys
    if explicit_nullable:
        schema["edgelabels"][0]["nullable_keys"] = ["at", "note"]
    clients = install_llm(monkeypatch, {TOKEN: [edge_graph(properties), PASS]})

    response = post(client, TOKEN, schema=schema, review_options={"max_fixes": 0})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "succeeded"
    chunk = data["chunk_results"][0]
    assert chunk["status"] == "final"
    assert chunk["issues"] == []
    assert (chunk["reviews_used"], chunk["fixes_used"], chunk["llm_calls"]) == (1, 0, 2)
    assert data["result"]["edges"][0]["properties"] == properties
    assert len(clients.clients[0].prompts) == 2
