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

import asyncio
import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
from mcp.types import CallToolResult


@pytest.fixture
def stdio_example(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples" / "stdio_check.py"
    spec = importlib.util.spec_from_file_location("stdio_check_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.list_tools.return_value = SimpleNamespace(
        tools=[SimpleNamespace(name="inspect_graph_tool"), SimpleNamespace(name="query_graph_data_tool")]
    )
    session.call_tool.return_value = CallToolResult(
        content=[], structuredContent={"ok": True, "data": {"hugegraph_server_status": "available"}}
    )

    @asynccontextmanager
    async def transport(_params):
        yield object(), object()

    monkeypatch.setattr(module, "stdio_client", transport)
    monkeypatch.setattr(module, "ClientSession", Mock(return_value=session))
    return module, session


@pytest.mark.parametrize(
    "invalid_request",
    [
        "{",
        "[]",
        "null",
        "42",
        '"tool"',
        "{}",
        '{"name": null}',
        '{"name": 42}',
        '{"name": ""}',
        '{"name": "  "}',
        '{"name": "query_graph_data_tool", "arguments": []}',
        '{"name": "query_graph_data_tool", "arguments": null}',
        '{"name": "query_graph_data_tool", "arguments": "query"}',
    ],
)
def test_interactive_invalid_input_keeps_same_session_open(stdio_example, monkeypatch, capsys, invalid_request):
    module, session = stdio_example
    monkeypatch.setattr(
        "builtins.input",
        Mock(
            side_effect=[
                invalid_request,
                '{"name": "query_graph_data_tool", "arguments": {"query": "g.V().count()"}}',
                EOFError,
            ]
        ),
    )

    asyncio.run(module.run(None, interactive=True))

    session.initialize.assert_awaited_once()
    module.ClientSession.assert_called_once()
    assert session.call_tool.await_args_list == [
        call("inspect_graph_tool", {"include_counts": False}),
        call("query_graph_data_tool", {"query": "g.V().count()"}),
    ]
    assert "Invalid tools/call request:" in capsys.readouterr().err


def test_interactive_blank_input_and_omitted_arguments(stdio_example, monkeypatch, capsys):
    module, session = stdio_example
    monkeypatch.setattr("builtins.input", Mock(side_effect=["  ", '{"name": "query_graph_data_tool"}', EOFError]))

    asyncio.run(module.run(None, interactive=True))

    assert session.call_tool.await_args_list == [
        call("inspect_graph_tool", {"include_counts": False}),
        call("query_graph_data_tool", {}),
    ]
    assert not capsys.readouterr().err


@pytest.mark.parametrize("failure", [ValueError("Tool execution failed"), CallToolResult(content=[], isError=True)])
def test_interactive_tool_failure_still_terminates(stdio_example, monkeypatch, capsys, failure):
    module, session = stdio_example
    session.call_tool.side_effect = [session.call_tool.return_value, failure]
    user_input = Mock(side_effect=['{"name": "query_graph_data_tool"}', EOFError])
    monkeypatch.setattr("builtins.input", user_input)

    error_type = ValueError if isinstance(failure, Exception) else RuntimeError
    with pytest.raises(error_type):
        asyncio.run(module.run(None, interactive=True))

    assert session.call_tool.await_count == 2
    user_input.assert_called_once()
    assert not capsys.readouterr().err
