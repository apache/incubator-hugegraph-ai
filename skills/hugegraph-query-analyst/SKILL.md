---
name: hugegraph-query-analyst
description: Route HugeGraph MCP structured graph query tasks to stable read-only tools. Use when the user asks graph questions, asks to generate Gremlin, provides Gremlin to explain, or needs read-only graph exploration.
---

# HugeGraph Query Analyst

## Tool Routes

| Goal | Tool |
| --- | --- |
| Inspect schema, labels, properties, or edge directions | `inspect_graph_tool(include_raw_schema=false)` or `inspect_graph_tool(include_raw_schema=true)` |
| Generate Gremlin from natural language | `generate_gremlin_tool(query, execute=false)` |
| Answer a natural-language graph question | Inspect schema, then use `query_graph_data_tool` for supported bounded queries |
| Execute user-provided Gremlin | Unavailable: Raw Gremlin execution returns `FEATURE_DISABLED`, including in admin mode |
| Generate only, without execution | `generate_gremlin_tool(query, execute=false)` |


## Order

```text
Need schema: inspect_graph_tool
Natural-language query: inspect_graph_tool -> query_graph_data_tool
Existing Gremlin: explain it; translate supported queries into query_graph_data_tool arguments
```

Structured query and plan lifecycle tools require the default `core` toolset. For a `legacy` deployment, enable `HUGEGRAPH_MCP_TOOLSET=core` and restart before using them.

These skills describe current source. Published `1.7.1` uses `v2_core` / `v1`; see the [migration guide](../../hugegraph-mcp/README.md#toolset-naming-and-migration).
