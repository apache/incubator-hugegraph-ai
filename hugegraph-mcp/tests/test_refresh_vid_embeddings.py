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

from unittest.mock import Mock

import pytest
import requests

from hugegraph_mcp.envelope import ErrorType, envelope_err, envelope_ok
from hugegraph_mcp.hugegraph_ai_client import get as ai_get
from hugegraph_mcp.hugegraph_ai_client import post as ai_post
from hugegraph_mcp.tools import refresh_vid_embeddings as refresh_vid_embeddings_module


@pytest.fixture(autouse=True)
def supported_refresh(monkeypatch):
    monkeypatch.setattr(refresh_vid_embeddings_module, "get", Mock(return_value=envelope_ok({"protocol_version": 1})))
    monkeypatch.setenv("HUGEGRAPH_GRAPH_PATH", "DEFAULT/hugegraph")


def test_refresh_vid_embeddings_success(monkeypatch):
    monkeypatch.setenv("HUGEGRAPH_MCP_READONLY", "false")
    post = Mock(return_value=envelope_ok({"ok": True, "data": {"added": 3, "removed": 1, "summary": "done"}}))
    monkeypatch.setattr(refresh_vid_embeddings_module, "post", post)

    result = refresh_vid_embeddings_module.refresh_vid_embeddings(confirm=True)

    assert result["ok"] is True
    assert result["data"] == {"added": 3, "removed": 1, "summary": "done"}
    assert post.call_args.args == ("/v1/vid-embeddings/refresh",)
    assert post.call_args.kwargs["json"] == {"expected_graph": "hugegraph", "expected_graphspace": "DEFAULT"}


def test_refresh_vid_embeddings_parses_remove_variants(monkeypatch):
    monkeypatch.setenv("HUGEGRAPH_MCP_READONLY", "false")
    post = Mock(return_value=envelope_ok("Remove 2 vectors, add 3 vectors."))
    monkeypatch.setattr(refresh_vid_embeddings_module, "post", post)

    result = refresh_vid_embeddings_module.refresh_vid_embeddings(confirm=True)

    assert result["ok"] is True
    assert result["data"]["added"] == 3
    assert result["data"]["removed"] == 2


def test_refresh_vid_embeddings_readonly(monkeypatch):
    monkeypatch.setenv("HUGEGRAPH_MCP_READONLY", "true")
    post = Mock()
    monkeypatch.setattr(refresh_vid_embeddings_module, "post", post)

    result = refresh_vid_embeddings_module.refresh_vid_embeddings(confirm=True)

    assert result["ok"] is False
    assert result["error"]["type"] == "READONLY_VIOLATION"
    post.assert_not_called()


def test_refresh_vid_embeddings_missing_confirm(monkeypatch):
    monkeypatch.setenv("HUGEGRAPH_MCP_READONLY", "false")
    post = Mock()
    monkeypatch.setattr(refresh_vid_embeddings_module, "post", post)

    result = refresh_vid_embeddings_module.refresh_vid_embeddings(confirm=False)

    assert result["ok"] is False
    assert result["error"]["type"] == "CONFIRM_REQUIRED"
    post.assert_not_called()


@pytest.mark.parametrize("data", [{}, {"protocol_version": True}, {"protocol_version": 2}, None])
def test_old_ai_never_receives_refresh(monkeypatch, data):
    monkeypatch.setenv("HUGEGRAPH_MCP_READONLY", "false")
    monkeypatch.setattr(refresh_vid_embeddings_module, "get", Mock(return_value=envelope_ok(data)))
    post = Mock()
    monkeypatch.setattr(refresh_vid_embeddings_module, "post", post)
    result = refresh_vid_embeddings_module.refresh_vid_embeddings(confirm=True)
    assert result["error"]["type"] == "FEATURE_DISABLED"
    post.assert_not_called()


@pytest.mark.parametrize(
    "error_type", [ErrorType.HUGEGRAPH_AI_UNAVAILABLE, ErrorType.AUTHORIZATION_FAILED, ErrorType.FEATURE_DISABLED]
)
def test_capability_errors_preserve_envelope_and_never_post(monkeypatch, error_type):
    monkeypatch.setenv("HUGEGRAPH_MCP_READONLY", "false")
    upstream = envelope_err(error_type, "probe failed", retryable=True, details={"reason": "test_failure"})
    monkeypatch.setattr(refresh_vid_embeddings_module, "get", Mock(return_value=upstream))
    post = Mock()
    monkeypatch.setattr(refresh_vid_embeddings_module, "post", post)
    assert refresh_vid_embeddings_module.refresh_vid_embeddings(confirm=True) is upstream
    post.assert_not_called()


@pytest.mark.parametrize("status_code", [401, 403, 404, 405, 429, 500, 502])
def test_http_capability_failures_keep_classification_without_refresh(monkeypatch, status_code):
    monkeypatch.setenv("HUGEGRAPH_MCP_READONLY", "false")
    monkeypatch.setenv("HUGEGRAPH_MCP_ALLOW_AI", "true")
    monkeypatch.setattr(refresh_vid_embeddings_module, "get", ai_get)
    response = requests.Response()
    response.status_code = status_code
    response.url = "http://ai.test/vid-embeddings/refresh-capabilities"
    response._content = b'{"detail":"probe failed"}'
    transport = Mock(return_value=response)
    monkeypatch.setattr(requests, "request", transport)
    result = refresh_vid_embeddings_module.refresh_vid_embeddings(confirm=True)
    assert result["ok"] is False
    expected = (
        "FEATURE_DISABLED"
        if status_code in {404, 405}
        else ("AUTHORIZATION_FAILED" if status_code in {401, 403} else "HUGEGRAPH_AI_UNAVAILABLE")
    )
    assert result["error"]["type"] == expected
    assert result["error"]["retryable"] is (status_code == 429 or status_code >= 500)
    assert result["error"]["details"]["status_code"] == status_code
    assert transport.call_count == 1
    assert transport.call_args.args[0] == "GET"


@pytest.mark.parametrize(
    "failure", [requests.exceptions.Timeout("timeout"), requests.exceptions.ConnectionError("offline")]
)
def test_network_capability_failure_remains_retryable_without_refresh(monkeypatch, failure):
    monkeypatch.setenv("HUGEGRAPH_MCP_READONLY", "false")
    monkeypatch.setenv("HUGEGRAPH_MCP_ALLOW_AI", "true")
    monkeypatch.setattr(refresh_vid_embeddings_module, "get", ai_get)
    transport = Mock(side_effect=failure)
    monkeypatch.setattr(requests, "request", transport)
    result = refresh_vid_embeddings_module.refresh_vid_embeddings(confirm=True)
    assert result["error"]["type"] == "HUGEGRAPH_AI_UNAVAILABLE"
    assert result["error"]["retryable"] is True
    assert transport.call_count == 1
    assert transport.call_args.args[0] == "GET"


@pytest.mark.parametrize("status_code", [404, 405])
def test_mixed_version_routing_never_dispatches_legacy_refresh(monkeypatch, status_code):
    monkeypatch.setenv("HUGEGRAPH_MCP_READONLY", "false")
    monkeypatch.setenv("HUGEGRAPH_MCP_ALLOW_AI", "true")
    monkeypatch.setenv("HUGEGRAPH_AI_URL", "http://ai.test")
    monkeypatch.setattr(refresh_vid_embeddings_module, "get", ai_get)
    monkeypatch.setattr(refresh_vid_embeddings_module, "post", ai_post)
    legacy_dispatch = Mock()

    def route_between_replicas(method, url, **kwargs):
        response = requests.Response()
        response.url = url
        response.status_code = 200
        if method == "GET":
            # Discovery reaches a new replica advertising the existing protocol.
            response._content = b'{"protocol_version":1}'
        elif url == "http://ai.test/vid-embeddings/refresh":
            # POST reaches an old replica: its empty request model ignores targets.
            legacy_dispatch(kwargs["json"])
            response._content = b'{"added":3,"removed":1}'
        else:
            # Old replicas cannot execute the versioned validated-refresh route.
            response.status_code = status_code
            response._content = b'{"detail":"Not Found"}'
        return response

    transport = Mock(side_effect=route_between_replicas)
    monkeypatch.setattr(requests, "request", transport)

    result = refresh_vid_embeddings_module.refresh_vid_embeddings(confirm=True)

    assert result["ok"] is False
    assert result["error"]["type"] == "FEATURE_DISABLED"
    assert result["error"]["details"]["status_code"] == status_code
    assert "upgrade" in result["error"]["suggestion"].lower()
    legacy_dispatch.assert_not_called()
    assert [(call.args[0], call.args[1]) for call in transport.call_args_list] == [
        ("GET", "http://ai.test/vid-embeddings/refresh-capabilities"),
        ("POST", "http://ai.test/v1/vid-embeddings/refresh"),
    ]


@pytest.mark.parametrize("status_code", [401, 403, 429, 500, 502])
def test_validated_refresh_errors_preserve_classification_without_fallback(monkeypatch, status_code):
    monkeypatch.setenv("HUGEGRAPH_MCP_READONLY", "false")
    monkeypatch.setenv("HUGEGRAPH_MCP_ALLOW_AI", "true")
    monkeypatch.setattr(refresh_vid_embeddings_module, "post", ai_post)
    response = requests.Response()
    response.status_code = status_code
    response._content = b'{"detail":"refresh failed"}'
    transport = Mock(return_value=response)
    monkeypatch.setattr(requests, "request", transport)

    result = refresh_vid_embeddings_module.refresh_vid_embeddings(confirm=True)

    expected = "AUTHORIZATION_FAILED" if status_code in {401, 403} else "HUGEGRAPH_AI_UNAVAILABLE"
    assert result["error"]["type"] == expected
    assert result["error"]["retryable"] is (status_code == 429 or status_code >= 500)
    assert transport.call_count == 1
    assert transport.call_args.args[0] == "POST"
    assert transport.call_args.args[1].endswith("/v1/vid-embeddings/refresh")
