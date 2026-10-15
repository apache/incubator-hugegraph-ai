---
name: hugegraph-regression-tester
description: Route HugeGraph MCP regression testing tasks to the current public tool surface, including inspection, read-only queries, Gremlin generation, extraction, controlled import, schema validate/dry-run, safety guards, and admin-gated tool checks.
---

# HugeGraph Regression Tester

## Tool Routes

| Test goal | Tool |
| --- | --- |
| MCP, graph status, or schema | `inspect_graph_tool(include_raw_schema=false)` and when needed `inspect_graph_tool(include_raw_schema=true)` |
| Structured read | `query_graph_data_tool` (bounded page or exact ID query) |
| Natural-language Gremlin generation | `generate_gremlin_tool(query, execute=false)` |
| Generate Gremlin without execution | `generate_gremlin_tool(query, execute=false)`; raw execution must return `FEATURE_DISABLED` |
| Raw Gremlin disabled boundary | `execute_gremlin_read_tool(unsafe_write_query)`; rejection verifies unconditional disablement, not write classification |
| Text-to-graph extraction | `extract_graph_data_tool(text, graph_schema?, example_prompt?)` |
| Import dry-run | `import_graph_data_tool(mode="ingest", graph_data, dry_run=true)` |
| Import preview boundary | Assert `confirmable=false`, `preview_only=true`, no `plan_id`; confirmation returns `FEATURE_DISABLED` |
| Verify existing graph data | `query_graph_data_tool` |
| Schema design | `design_schema_tool(operations?)` |
| Schema validation | `apply_schema_tool(mode="validate", operations)` |
| Schema dry-run | `apply_schema_tool(mode="dry_run", operations)` |
| Raw write Gremlin disabled boundary | `execute_gremlin_write_tool(gremlin_query)`; must remain disabled even in admin mode |
| Admin gate check without a write | `refresh_vid_embeddings_tool(confirm=false)` in separate admin-disabled and admin-enabled processes |

## Order

```text
inspect -> generate/read -> extract -> import dry-run -> assert preview-only boundary
-> verify read -> schema validate/dry-run -> safety/gate checks
```

Structured query and plan lifecycle tools require the default `core` toolset. For a `legacy` deployment, enable `HUGEGRAPH_MCP_TOOLSET=core` and restart before using them.

With `HUGEGRAPH_MCP_READONLY=false`, the admin-disabled refresh call must return
`FEATURE_DISABLED` with details requiring `HUGEGRAPH_MCP_ADMIN_MODE=true`; the
admin-enabled call must return `CONFIRM_REQUIRED`. Keep `confirm=false` in both
processes to avoid a refresh. Restart between configuration changes. These calls
do not verify AI-side authentication or target validation. Validate the write
classifier independently with unit tests; raw Gremlin rejection cannot exercise it.
For the refresh contract, run the API/MCP regression tests covering default
GraphSpace equivalence, required fields and write/auth gates on
`POST /v1/vid-embeddings/refresh`, configuration locking, and mixed-version routing.
The capability GET is only discovery; MCP must never fall back to the legacy POST
when the versioned route is unavailable. This does not verify physical-instance
identity or establish a live rolling-deployment test.
Generation and extraction require `HUGEGRAPH_MCP_ALLOW_AI=true` and a configured AI service.
These skills describe current source; published `1.7.1` uses `v2_core` / `v1`.
