# Output 目录

这个目录用于存放生成的 Gremlin 查询语料库文件。

## 文件命名规则

生成的文件会自动命名为：
```
generated_corpus_YYYYMMDD_HHMMSS.json
```

例如：
- `generated_corpus_20251029_143025.json`
- `generated_corpus_20251029_150130.json`

## 文件格式

每个生成的 JSON 文件包含：

```json
{
    "metadata": {
        "total_templates": 10,
        "successful_templates": 9,
        "failed_templates": 1,
        "total_unique_queries": 450,
        "generation_timestamp": "2025-10-29 14:30:25"
    },
    "corpus": [
        {
            "query": "g.V().hasLabel('person').has('name', 'Tom Hanks')",
            "description": "从图中开始，并筛选出标签为 'person' 的元素，并筛选出属性 'name' 为 'Tom Hanks' 的元素"
        }
    ]
}
```

## 使用方式

### 生成新的语料库

```bash
# 使用默认配置
python generate_corpus.py

# 指定生成数量
python generate_corpus.py --num-queries 50
```

### 查看生成的文件

```bash
# 列出所有生成的文件
ls -lh output/

# 查看最新生成的文件
ls -t output/ | head -1
```

### 清理旧文件

```bash
# 删除所有生成的文件
rm output/generated_corpus_*.json

# 只保留最新的 5 个文件
ls -t output/generated_corpus_*.json | tail -n +6 | xargs rm
```

## 注意事项

- 每次运行 `generate_corpus.py` 都会生成一个新文件
- 文件不会自动覆盖，需要手动清理旧文件
- 建议定期清理不需要的文件以节省空间
