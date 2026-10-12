---
name: hugegraph-schema-designer
description: Route HugeGraph MCP schema design, validation, and dry-run preview tasks to stable public tools. Use when the user asks to design vertex labels, edge labels, properties, primary keys, indexes, or preview schema operations.
---

# HugeGraph Schema Designer

## Tool Routes

| Goal | Tool |
| --- | --- |
| Inspect current schema | `inspect_graph_tool(include_raw_schema=true)` |
| Design schema operations | `design_schema_tool(operations?)` |
| Validate schema operations | `apply_schema_tool(mode="validate", operations)` |
| Dry-run schema operations | `apply_schema_tool(mode="dry_run", operations)` |
| Execute real schema apply | `confirm_write_tool(plan_id)` in default `core`, one supported schema create per plan |
| Delete or roll back schema | No V1 stable tool |

The default `core` toolset permits confirmed apply only for
`create_property_key`, `create_vertex_label`, and `create_edge_label`. In
`HUGEGRAPH_MCP_TOOLSET=legacy`, real schema apply remains disabled.

Review the dry-run response before applying it. Confirm only a returned, confirmable
`plan_id` with `confirm_write_tool(plan_id)`, then query `get_write_status_tool(plan_id)`.

Schema create operations use a closed field contract. Property keys accept
`data_type`, `cardinality`, `aggregate_type`, and `user_data`; vertex labels
accept `id_strategy`, `properties`, `primary_keys`, `nullable_keys`,
`index_labels`, `enable_label_index`, and `user_data`; edge labels accept
`source_label`, `target_label`, `properties`, `nullable_keys`, `sort_keys`,
`frequency`, `enable_label_index`, and `user_data`. Unsupported fields,
including `ttl*`, are rejected during validation instead of being silently
dropped. Supported fields are forwarded and checked by a live-schema post-read.

## Order

```text
inspect_graph_tool -> design_schema_tool
-> apply_schema_tool(mode="validate")
-> apply_schema_tool(mode="dry_run")
-> review operations and preview
-> confirm_write_tool(plan_id) -> get_write_status_tool(plan_id)
```

Structured query and plan lifecycle tools require the default `core` toolset. For a `legacy` deployment, enable `HUGEGRAPH_MCP_TOOLSET=core` and restart before using them.

These skills describe current source. Published `1.7.1` uses `v2_core` / `v1`; see the [migration guide](../../hugegraph-mcp/README.md#toolset-naming-and-migration).
