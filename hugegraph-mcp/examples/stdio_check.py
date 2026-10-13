# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch one STDIO session, inspect it, then execute optional JSON tool requests.

Usage: python hugegraph-mcp/examples/stdio_check.py [requests.json]
requests.json is a list of {"name": ..., "arguments": ...} tools/call payloads.
The same session stays open for all calls. Server inherits the caller environment.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run(requests_path: Path | None, interactive: bool):
    requests = json.loads(requests_path.read_text(encoding="utf-8")) if requests_path else []
    if not isinstance(requests, list):
        raise ValueError("Requests must be a JSON list")
    params = StdioServerParameters(command=sys.executable, args=["-m", "hugegraph_mcp.server"], env=dict(os.environ))
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listed = await session.list_tools()
        sys.stdout.write(json.dumps({"tools": [tool.name for tool in listed.tools]}) + "\n")
        pending = [{"name": "inspect_graph_tool", "arguments": {"include_counts": False}}, *requests]
        while True:
            if pending:
                request = pending.pop(0)
            elif interactive:
                try:
                    line = await asyncio.to_thread(input, "tools/call JSON (EOF to finish): ")
                except EOFError:
                    break
                if not line.strip():
                    continue
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise ValueError("Request must be a JSON object")
                    if not isinstance(request.get("name"), str) or not request["name"].strip():
                        raise ValueError("Tool name must be a non-empty string")
                    if not isinstance(request.get("arguments", {}), dict):
                        raise ValueError("Tool arguments must be a JSON object")
                except ValueError as exc:
                    sys.stderr.write(f"Invalid tools/call request: {exc}\n")
                    continue
            else:
                break
            result = await session.call_tool(request["name"], request.get("arguments", {}))
            sys.stdout.write(result.model_dump_json() + "\n")
            if result.isError:
                raise RuntimeError(f"MCP tool failed: {request['name']}")
            data = result.structuredContent
            if data is None:
                data = json.loads(result.content[0].text)
            if request["name"] == "inspect_graph_tool" and (
                not data.get("ok") or data["data"].get("hugegraph_server_status") != "available"
            ):
                raise RuntimeError("Graph inspection could not verify connection; inspect warnings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", nargs="?", type=Path)
    parser.add_argument("--interactive", action="store_true", help="Read one tools/call JSON object per line")
    args = parser.parse_args()
    asyncio.run(run(args.requests, args.interactive))


if __name__ == "__main__":
    main()
