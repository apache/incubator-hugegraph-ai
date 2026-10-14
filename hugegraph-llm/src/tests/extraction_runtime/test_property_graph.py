# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from copy import deepcopy

import pytest

from hugegraph_llm.config import prompt
from hugegraph_llm.extraction_runtime.property_graph import PropertyGraphBundle
from hugegraph_llm.extraction_runtime.v1 import ChunkInput, ExtractionEngineV1, ExtractionOptions, Issue

pytestmark = pytest.mark.contract
CHUNK = ChunkInput(index=0, text="Alice works for Acme. Alice is 30 years old.")


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate(self, *, prompt):
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return json.dumps(response) if not isinstance(response, str) else response


@pytest.fixture
def schema():
    return {
        "vertexlabels": [
            {
                "id": 1,
                "name": "person",
                "id_strategy": "PRIMARY_KEY",
                "primary_keys": ["name"],
                "properties": ["name", "age", "tags"],
                "nullable_keys": ["age", "tags"],
            },
            {"id": 2, "name": "company", "primary_keys": ["name"], "properties": ["name"], "nullable_keys": []},
        ],
        "edgelabels": [{"name": "works_for", "source_label": "person", "target_label": "company", "properties": []}],
        "propertykeys": [
            {"name": "name", "data_type": "TEXT", "cardinality": "SINGLE"},
            {"name": "age", "data_type": "INT", "cardinality": "SINGLE"},
            {"name": "tags", "data_type": "TEXT", "cardinality": "SET"},
        ],
    }


@pytest.fixture
def graph():
    return {
        "vertices": [
            {"id": "alice", "label": "person", "properties": {"name": "Alice", "age": 30}},
            {"id": "acme", "label": "company", "properties": {"name": "Acme"}},
        ],
        "edges": [
            {
                "label": "works_for",
                "outV": "alice",
                "outVLabel": "person",
                "inV": "acme",
                "inVLabel": "company",
                "properties": {},
            }
        ],
    }


def bundle(schema, *responses):
    return PropertyGraphBundle(llm=ScriptedLLM(*responses), schema=schema)


def test_extract_and_validate_normalize_ids_without_changing_schema(schema, graph):
    original = deepcopy(schema)
    domain = bundle(schema, "```json\n" + json.dumps(graph) + "\n```")
    parsed = domain.extract(CHUNK)
    assert domain.validate(parsed, CHUNK) == []
    assert parsed["vertices"][0]["id"] == "1:Alice"
    assert parsed["edges"][0]["outV"] == "1:Alice"
    assert parsed["edges"][0]["inV"] == "2:Acme"
    assert parsed["vertices"][0]["type"] == "vertex"
    normalized = deepcopy(parsed)
    assert domain.validate(parsed, CHUNK) == []
    assert parsed == normalized
    assert schema == original
    assert CHUNK.text in domain.llm.prompts[0]
    assert prompt.extract_graph_prompt in domain.llm.prompts[0]


def test_legacy_endpoint_and_multiple_primary_keys(schema, graph):
    schema["vertexlabels"][0]["primary_keys"] = ["name", "age"]
    graph["edges"][0] = {
        "label": "works_for",
        "source": {"label": "person", "properties": {"name": "Alice", "age": 30}},
        "target": {"label": "company", "properties": {"name": "Acme"}},
        "properties": {},
    }
    domain = bundle(schema, graph)
    parsed = domain.extract(CHUNK)
    assert domain.validate(parsed, CHUNK) == []
    assert parsed["edges"][0]["outV"] == "1:Alice!30"
    assert parsed["edges"][0]["inV"] == "2:Acme"


def test_invalid_entities_and_edges_remain_available_for_repair(schema, graph):
    graph["vertices"].append({"label": "unknown", "properties": {"unexpected": 3}})
    graph["vertices"].append("not-an-object")
    graph["vertices"][0]["properties"]["surprise"] = True
    graph["edges"][0]["inV"] = "missing"
    domain = bundle(schema, graph)
    parsed = domain.extract(CHUNK)
    issues = domain.validate(parsed, CHUNK)
    assert len(parsed["vertices"]) == 4 and len(parsed["edges"]) == 1
    assert parsed["vertices"][0]["properties"]["surprise"] is True
    assert parsed["edges"][0]["inV"] == "missing"
    assert {issue.path for issue in issues} >= {
        "vertices[2].label",
        "vertices[3]",
        "vertices[0].properties.surprise",
        "edges[0].inV",
    }


@pytest.mark.parametrize("properties", [{}, None, [], "bad"])
def test_missing_or_invalid_properties_are_reported(schema, graph, properties):
    graph["vertices"][0]["properties"] = properties
    domain = bundle(schema)
    issues = domain.validate(graph, CHUNK)
    assert any(issue.path.startswith("vertices[0].properties") for issue in issues)
    assert graph["vertices"][0]["properties"] == properties


@pytest.mark.parametrize("value", ["30", True, 2**31, 3.2])
def test_integer_type_and_range_are_checked_without_coercion(schema, graph, value):
    graph["vertices"][0]["properties"]["age"] = value
    issues = bundle(schema).validate(graph, CHUNK)
    assert any(issue.path == "vertices[0].properties.age" for issue in issues)
    assert graph["vertices"][0]["properties"]["age"] == value


@pytest.mark.parametrize("value", ["engineer", ["engineer", "engineer"], ["engineer", 3]])
def test_collection_cardinality_and_element_types(schema, graph, value):
    graph["vertices"][0]["properties"]["tags"] = value
    assert any(issue.path == "vertices[0].properties.tags" for issue in bundle(schema).validate(graph, CHUNK))


def test_missing_schema_constraints_are_not_invented(graph):
    schema = {
        "vertexlabels": [{"name": "person"}, {"name": "company"}],
        "edgelabels": [{"name": "works_for"}],
    }
    graph["vertices"][0]["properties"] = {}
    domain = bundle(schema)
    assert domain.validate(graph, CHUNK) == []
    assert graph["vertices"][0]["id"] == "alice"


def test_duplicate_ids_primary_keys_and_aliases_remain_invalid_on_revalidation(schema, graph):
    graph["vertices"].append(deepcopy(graph["vertices"][0]))
    domain = bundle(schema)
    first = domain.validate(graph, CHUNK)
    assert any("Duplicate vertex id" in issue.message for issue in first)
    assert any("Duplicate vertex primary key" in issue.message for issue in first)
    assert domain.validate(graph, CHUNK) == first
    graph["vertices"][-1]["properties"]["name"] = "Bob"
    before = deepcopy(graph)
    first = domain.validate(graph, CHUNK)
    assert any("Ambiguous" in issue.message for issue in first)
    assert graph == before
    assert domain.validate(graph, CHUNK) == first


def test_existing_endpoint_with_wrong_label_and_direction_is_reported(schema, graph):
    graph["edges"][0].update(outV="acme", outVLabel="person", inV="alice", inVLabel="person")
    issues = bundle(schema).validate(graph, CHUNK)
    assert any(issue.path == "edges[0].outV" for issue in issues)
    assert any(issue.path == "edges[0].inVLabel" and "schema" in issue.message for issue in issues)
    assert len(graph["edges"]) == 1


def test_canonical_endpoint_id_must_have_its_actual_vertex_label(schema, graph):
    graph["edges"][0].update(outV="2:Acme", outVLabel="person")
    issues = bundle(schema).validate(graph, CHUNK)
    assert any(issue.path == "edges[0].outVLabel" and "referenced vertex" in issue.message for issue in issues)


def test_custom_id_strategy_is_preserved_and_checked(schema, graph):
    schema["vertexlabels"][0]["id_strategy"] = "CUSTOMIZE_NUMBER"
    del schema["vertexlabels"][0]["primary_keys"]
    graph["vertices"][0]["id"] = 0
    graph["edges"][0]["outV"] = 0
    domain = bundle(schema)
    assert domain.validate(graph, CHUNK) == []
    assert graph["vertices"][0]["id"] == 0 and graph["edges"][0]["outV"] == 0
    graph["vertices"][0]["id"] = "not numeric"
    assert any("CUSTOMIZE_NUMBER" in issue.message for issue in domain.validate(graph, CHUNK))


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "{}",
        '{"vertices": [], "edges": null}',
        "[{}]",
        "NaN",
        '{"vertices": [], "edges": [], "value": 1e999}',
    ],
)
def test_invalid_graph_response_fails_instead_of_becoming_empty(schema, response):
    domain = bundle(schema, response)
    result = ExtractionEngineV1().run(bundle=domain, chunk=CHUNK, options=ExtractionOptions())
    assert result.status == "failed" and result.stage == "extract"
    assert result.graph is None and result.llm_calls == 1


def test_flat_array_format_and_legitimate_empty_graph(schema):
    domain = bundle(schema, [], {"decision": "pass", "issues": []})
    result = ExtractionEngineV1().run(bundle=domain, chunk=CHUNK, options=ExtractionOptions())
    assert result.status == "final" and result.graph == {"vertices": [], "edges": []}
    assert result.llm_calls == 2


@pytest.mark.parametrize(
    "response",
    [
        {"decision": "unknown", "issues": []},
        {"decision": "pass"},
        {"decision": "pass", "issues": [{"path": "graph", "message": "wrong"}]},
        {"decision": "fix", "issues": []},
        {"decision": "block", "issues": ["wrong"]},
        {"decision": "fix", "issues": [{"path": "", "message": "wrong"}]},
        {"decision": "fix", "issues": [{"path": "graph", "message": "wrong", "evidence": 3}]},
    ],
)
def test_malformed_review_is_a_failed_review(schema, graph, response):
    result = ExtractionEngineV1().run(bundle=bundle(schema, graph, response), chunk=CHUNK, options=ExtractionOptions())
    assert result.status == "failed" and result.stage == "review"
    assert result.graph is not None and result.llm_calls == 2


def test_review_repair_full_graph_flow_and_prompt_context(schema, graph, monkeypatch):
    monkeypatch.setattr(prompt, "review_graph_prompt", "CUSTOM REVIEW")
    monkeypatch.setattr(prompt, "repair_graph_prompt", "CUSTOM REPAIR")
    wrong = deepcopy(graph)
    wrong["vertices"][0]["properties"]["age"] = 40
    review = {
        "decision": "fix",
        "issues": [{"path": "vertices[0].properties.age", "message": "Age should be 30", "evidence": CHUNK.text}],
    }
    domain = bundle(schema, wrong, review, graph, {"decision": "pass", "issues": []})
    result = ExtractionEngineV1().run(bundle=domain, chunk=CHUNK, options=ExtractionOptions())
    assert result.status == "final" and result.graph["vertices"][0]["properties"]["age"] == 30
    assert result.issues == [] and (result.reviews_used, result.fixes_used, result.llm_calls) == (2, 1, 4)
    review_context = json.loads(domain.llm.prompts[1].split("\n## Input\n")[1])
    repair_context = json.loads(domain.llm.prompts[2].split("\n## Input\n")[1])
    assert domain.llm.prompts[1].startswith("CUSTOM REVIEW")
    assert domain.llm.prompts[2].startswith("CUSTOM REPAIR")
    assert review_context["text"] == CHUNK.text and review_context["schema"] == schema
    assert repair_context["graph"]["vertices"][0]["properties"]["age"] == 40
    assert repair_context["issues"] == review["issues"]


def test_schema_issues_enter_repair_and_repaired_graph_is_revalidated(schema, graph):
    wrong = deepcopy(graph)
    wrong["vertices"][0]["properties"]["age"] = "thirty"
    also_wrong = deepcopy(graph)
    also_wrong["edges"][0]["inV"] = "missing"
    domain = bundle(schema, wrong, also_wrong)
    result = ExtractionEngineV1().run(bundle=domain, chunk=CHUNK, options=ExtractionOptions(max_fixes=1))
    assert result.status == "blocked" and result.reviews_used == 0
    assert any(issue.path == "edges[0].inV" for issue in result.issues)
    assert not any(issue.path == "vertices[0].properties.age" for issue in result.issues)
    repair_context = json.loads(domain.llm.prompts[1].split("\n## Input\n")[1])
    assert repair_context["graph"]["vertices"][0]["properties"]["age"] == "thirty"
    assert any(issue["path"] == "vertices[0].properties.age" for issue in repair_context["issues"])


def test_failed_repair_preserves_previous_graph(schema, graph):
    domain = bundle(
        schema, graph, {"decision": "fix", "issues": [{"path": "edges", "message": "Wrong relation"}]}, "bad JSON"
    )
    result = ExtractionEngineV1().run(bundle=domain, chunk=CHUNK, options=ExtractionOptions())
    assert result.status == "failed" and result.stage == "repair"
    assert result.graph["edges"][0]["label"] == "works_for"
    assert result.llm_calls == 3 and result.fixes_used == 1


def test_custom_extract_prompt_and_isolated_schema(schema, graph):
    domain = PropertyGraphBundle(llm=ScriptedLLM(graph), schema=schema, example_prompt="CUSTOM EXTRACT\n")
    schema["vertexlabels"].clear()
    parsed = domain.extract(CHUNK)
    assert domain.llm.prompts[0].startswith("CUSTOM EXTRACT\n")
    assert domain.validate(parsed, CHUNK) == []


def test_explicit_block_review_keeps_evidence(schema, graph):
    review = {
        "decision": "block",
        "issues": [{"path": "graph", "message": "Not supported by source", "evidence": "source"}],
    }
    result = ExtractionEngineV1().run(bundle=bundle(schema, graph, review), chunk=CHUNK, options=ExtractionOptions())
    assert result.status == "blocked"
    assert result.issues == [Issue("graph", "Not supported by source", "source")]
