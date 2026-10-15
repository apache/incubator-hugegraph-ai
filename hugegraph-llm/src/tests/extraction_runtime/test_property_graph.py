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


@pytest.mark.parametrize(
    "data_type,value",
    [
        ("DATE", "2024-02-29"),
        ("DATE", "2024-2-9"),
        ("DATE", "2024-02-29 12:34:56"),
        ("DATE", "2024-2-9 12:34:56.1"),
        ("DATE", "2024-02-29 12:34:56.12"),
        ("DATE", "2024-02-29 12:34:56.123"),
        ("DATE", 1709208000000),
        ("DATE", 0),
        ("DATE", -1),
        ("DATE", -(2**63)),
        ("DATE", 2**63 - 1),
        ("OBJECT", {"source": "invoice", "items": [1, 2]}),
        ("OBJECT", [{"source": "invoice"}]),
        ("OBJECT", "invoice"),
        ("BLOB", "AQID"),
        ("BLOB", "0x010203"),
        ("BLOB", [1, 2, 3]),
        ("UNMODELED_TYPE", {"value": 1}),
    ],
)
def test_compatible_property_values_reach_final_without_repair(schema, graph, data_type, value):
    schema["propertykeys"][1]["data_type"] = data_type
    graph["vertices"][0]["properties"]["age"] = value
    domain = bundle(schema, graph, {"decision": "pass", "issues": []})

    assert domain.validate(deepcopy(graph), CHUNK) == []
    result = ExtractionEngineV1().run(bundle=domain, chunk=CHUNK, options=ExtractionOptions())

    assert result.status == "final" and result.issues == []
    assert (result.reviews_used, result.fixes_used, result.llm_calls) == (1, 0, 2)
    assert result.graph["vertices"][0]["properties"] == graph["vertices"][0]["properties"]
    review_context = json.loads(domain.llm.prompts[1].split("\n## Input\n")[1])
    assert review_context["graph"]["vertices"][0]["properties"]["age"] == value


@pytest.mark.parametrize(
    "value",
    [
        "2023-02-29",
        "2024-02-30 12:34:56",
        "2024-13-01",
        "0000-01-01",
        "2024-02-29 24:00:00",
        "2024-02-29 12:60:00",
        "2024-02-29 12:34:60",
        "2024-02-29 1:34:56",
        "2024-02-29 12:34:56.1234",
        "2024-02-29T12:34:56Z",
        "2024-02-29 12:34:56 trailing",
        "1709208000000",
        "not-a-date",
        True,
        False,
        1709208000000.0,
        2**63,
        -(2**63) - 1,
        ["2024-02-29"],
        {"date": "2024-02-29"},
    ],
)
def test_invalid_dates_are_blocked_before_review_and_preserved(schema, graph, value):
    schema["propertykeys"][1]["data_type"] = "DATE"
    graph["vertices"][0]["properties"]["age"] = value
    result = ExtractionEngineV1().run(bundle=bundle(schema, graph), chunk=CHUNK, options=ExtractionOptions(max_fixes=0))
    assert result.status == "blocked" and result.reason == "validation_failed"
    assert (result.reviews_used, result.fixes_used, result.llm_calls) == (0, 0, 1)
    assert any(issue.path == "vertices[0].properties.age" for issue in result.issues)
    assert result.graph["vertices"][0]["properties"]["age"] == value


@pytest.mark.parametrize("data_type,value", [("TEXT", ["x"]), ("BOOLEAN", {}), ("DOUBLE", []), ("UUID", {})])
def test_modeled_single_properties_still_reject_containers(schema, graph, data_type, value):
    schema["propertykeys"][1]["data_type"] = data_type
    graph["vertices"][0]["properties"]["age"] = value
    assert any(issue.path == "vertices[0].properties.age" for issue in bundle(schema).validate(graph, CHUNK))


@pytest.mark.parametrize(
    "data_type,cardinality,value",
    [
        ("DATE", "LIST", ["2024-02-29 12:34:56", 0, "2024-02-29 12:34:56"]),
        ("DATE", "SET", ["2024-02-29", 0]),
        ("OBJECT", "LIST", [{"value": 1}, {"value": 1}]),
        ("OBJECT", "SET", [{"value": 1}, {"value": 2}]),
        ("BLOB", "LIST", [[1, 2], [1, 2]]),
        ("BLOB", "SET", [[1, 2], [3, 4]]),
    ],
)
def test_compatible_collection_properties(schema, graph, data_type, cardinality, value):
    schema["propertykeys"][1].update(data_type=data_type, cardinality=cardinality)
    graph["vertices"][0]["properties"]["age"] = value
    assert bundle(schema).validate(graph, CHUNK) == []


@pytest.mark.parametrize(
    "data_type,cardinality,value,message",
    [
        ("DATE", "LIST", "2024-02-29", "Expected an array"),
        ("DATE", "LIST", ["2024-02-30"], "data type DATE"),
        ("OBJECT", "LIST", {"value": 1}, "Expected an array"),
        ("OBJECT", "SET", [{"value": 1}, {"value": 1}], "contains duplicates"),
        ("BLOB", "SET", "AQID", "Expected an array"),
        ("BLOB", "SET", [[1, 2], [1, 2]], "contains duplicates"),
        ("OBJECT", "LIST", [None], "data type OBJECT"),
        ("BLOB", "LIST", [None], "data type BLOB"),
    ],
)
def test_collection_constraints_still_apply(schema, graph, data_type, cardinality, value, message):
    schema["propertykeys"][1].update(data_type=data_type, cardinality=cardinality)
    graph["vertices"][0]["properties"]["age"] = value
    issues = bundle(schema).validate(graph, CHUNK)
    assert any(issue.path == "vertices[0].properties.age" and message in issue.message for issue in issues)
    assert graph["vertices"][0]["properties"]["age"] == value


@pytest.mark.parametrize("data_type", ["DATE", "OBJECT", "BLOB"])
@pytest.mark.parametrize("required", [False, True])
def test_null_properties_remain_invalid_for_all_types(schema, graph, data_type, required):
    schema["propertykeys"][1]["data_type"] = data_type
    if required:
        schema["vertexlabels"][0]["nullable_keys"].remove("age")
    graph["vertices"][0]["properties"]["age"] = None
    issues = bundle(schema).validate(graph, CHUNK)
    expected = "Required property" if required else "Omit absent properties"
    assert any(issue.path == "vertices[0].properties.age" and expected in issue.message for issue in issues)


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


def _set_uuid_case(schema, graph, target, value):
    if target == "property":
        schema["propertykeys"][1]["data_type"] = "UUID"
        graph["vertices"][0]["properties"]["age"] = value
        return "vertices[0].properties.age"
    schema["vertexlabels"][0]["id_strategy"] = "CUSTOMIZE_UUID"
    del schema["vertexlabels"][0]["primary_keys"]
    graph["vertices"][0]["id"] = value
    graph["edges"][0]["outV"] = value
    return "vertices[0].id"


def _uuid_case_value(graph, target):
    vertex = graph["vertices"][0]
    return vertex["properties"]["age"] if target == "property" else vertex["id"]


@pytest.mark.parametrize("target", ["property", "id"])
@pytest.mark.parametrize(
    "value",
    [
        "12345678-1234-5678-1234-567812345678",
        "ABCDEFAB-CDEF-ABCD-EFAB-CDEFABCDEFAB",
        "12345678123456781234567812345678",
        "ABCDEFABCDEFABCDEFABCDEFABCDEFAB",
        "00000000-0000-0000-0000-000000000000",
        "ffffffffffffffffffffffffffffffff",
        # StringEncoding.uuid delegates these 36-character forms to Java UUID.fromString.
        "123456789-123-1234-1234-123456789012",
        "+1234567-1234-1234-1234-123456789012",
        "00000000000000001-1-1-1-000000000000",
        "7fffffffffffffff-1-1-1-0000000000000",
        "+123456789abcdef+123456789abcdef",
        "ＡＢＣＤＥＦ12-1234-1234-1234-123456789012",
        "١٢٣٤٥٦٧٨-1234-1234-1234-123456789012",
    ],
)
def test_hugegraph_compatible_uuids_reach_final_without_normalization(schema, graph, target, value):
    _set_uuid_case(schema, graph, target, value)
    domain = bundle(schema, graph, {"decision": "pass", "issues": []})
    result = ExtractionEngineV1().run(bundle=domain, chunk=CHUNK, options=ExtractionOptions())

    assert result.status == "final" and result.issues == []
    assert (result.reviews_used, result.fixes_used, result.llm_calls) == (1, 0, 2)
    assert _uuid_case_value(result.graph, target) == value
    if target == "id":
        assert result.graph["edges"][0]["outV"] == value
    review_graph = json.loads(domain.llm.prompts[1].split("\n## Input\n")[1])["graph"]
    assert _uuid_case_value(review_graph, target) == value


@pytest.mark.parametrize("target", ["property", "id"])
@pytest.mark.parametrize(
    "value",
    [
        "not-a-uuid",
        "",
        None,
        123,
        True,
        {},
        ["12345678-1234-5678-1234-567812345678"],
        "12345678-1234-5678-1234-56781234567g",
        "1234567812345678123456781234567g",
        "1-1-1-1-1",
        "123456781234567812345678123456789",
        "12345678123456781234567812345678901",
        "12345678_1234-5678-1234-567812345678",
        "12345678--234-5678-1234-567812345678",
        "12345678-1234-5678-1234-56781234567 ",
        "1234567_123456781234567812345678",
        "{12345678-1234-5678-1234-567812345678}",
        "urn:uuid:12345678-1234-5678-1234-567812345678",
        # Java's signed Long parser rejects overflow despite the total length being 36.
        "8000000000000000-1-1-1-0000000000000",
        # Python accepts these decimal digits, but Java parses UTF-16 chars, not code points.
        "𝟙2345678-1234-5678-1234-567812345678",
    ],
)
def test_invalid_uuids_are_blocked_before_review_and_preserved(schema, graph, target, value):
    path = _set_uuid_case(schema, graph, target, value)
    result = ExtractionEngineV1().run(bundle=bundle(schema, graph), chunk=CHUNK, options=ExtractionOptions(max_fixes=0))

    assert result.status == "blocked" and result.reason == "validation_failed"
    assert (result.reviews_used, result.fixes_used, result.llm_calls) == (0, 0, 1)
    assert any(issue.path == path for issue in result.issues)
    assert _uuid_case_value(result.graph, target) == value


@pytest.mark.parametrize("cardinality", ["LIST", "SET"])
@pytest.mark.parametrize("valid", [False, True])
def test_uuid_collection_elements_are_validated(schema, graph, cardinality, valid):
    value = ["12345678-1234-5678-1234-567812345678", "ABCDEFABCDEFABCDEFABCDEFABCDEFAB" if valid else "not-a-uuid"]
    _set_uuid_case(schema, graph, "property", value)
    schema["propertykeys"][1]["cardinality"] = cardinality
    issues = bundle(schema).validate(graph, CHUNK)
    assert (issues == []) is valid
    if not valid:
        assert any(issue.path == "vertices[0].properties.age" and "UUID" in issue.message for issue in issues)
    assert graph["vertices"][0]["properties"]["age"] == value


@pytest.mark.parametrize("target", ["property", "id"])
def test_invalid_uuid_can_be_repaired_then_reviewed(schema, graph, target):
    valid = "12345678-1234-5678-1234-567812345678"
    _set_uuid_case(schema, graph, target, valid)
    wrong = deepcopy(graph)
    if target == "id":
        wrong["vertices"][0]["id"] = wrong["edges"][0]["outV"] = "not-a-uuid"
    else:
        wrong["vertices"][0]["properties"]["age"] = "not-a-uuid"
    domain = bundle(schema, wrong, graph, {"decision": "pass", "issues": []})
    result = ExtractionEngineV1().run(bundle=domain, chunk=CHUNK, options=ExtractionOptions())

    assert result.status == "final" and result.issues == []
    assert _uuid_case_value(result.graph, target) == valid
    assert (result.reviews_used, result.fixes_used, result.llm_calls) == (1, 1, 3)
    repair_context = json.loads(domain.llm.prompts[1].split("\n## Input\n")[1])
    assert _uuid_case_value(repair_context["graph"], target) == "not-a-uuid"
    assert any("UUID" in issue["message"] for issue in repair_context["issues"])


@pytest.mark.parametrize("target", ["property", "id"])
def test_uuid_repair_is_revalidated_and_retains_latest_invalid_graph(schema, graph, target):
    _set_uuid_case(schema, graph, target, "not-a-uuid")
    replacement = deepcopy(graph)
    if target == "id":
        replacement["vertices"][0]["id"] = replacement["edges"][0]["outV"] = "still-not-a-uuid"
    else:
        replacement["vertices"][0]["properties"]["age"] = "still-not-a-uuid"
    result = ExtractionEngineV1().run(
        bundle=bundle(schema, graph, replacement), chunk=CHUNK, options=ExtractionOptions(max_fixes=1)
    )

    assert result.status == "blocked" and result.reason == "validation_failed"
    assert _uuid_case_value(result.graph, target) == "still-not-a-uuid"
    assert any("UUID" in issue.message for issue in result.issues)
    assert (result.reviews_used, result.fixes_used, result.llm_calls) == (0, 1, 2)


@pytest.mark.parametrize("target", ["property", "id"])
def test_failed_uuid_repair_keeps_invalid_value_and_diagnostics(schema, graph, target):
    path = _set_uuid_case(schema, graph, target, "not-a-uuid")
    result = ExtractionEngineV1().run(
        bundle=bundle(schema, graph, "bad JSON"), chunk=CHUNK, options=ExtractionOptions()
    )

    assert result.status == "failed" and result.stage == "repair"
    assert result.reason == "repair_failed: JSONDecodeError"
    assert _uuid_case_value(result.graph, target) == "not-a-uuid"
    assert any(issue.path == path and "UUID" in issue.message for issue in result.issues)
    assert (result.reviews_used, result.fixes_used, result.llm_calls) == (0, 1, 2)
