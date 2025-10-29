# Gremlin 查询语料库生成器

从模板生成大量多样化的 Gremlin 查询及其中文描述，用于测试、训练和分析。

## 快速开始
环境配置：python：3.12.10
```bash
pip install requirements.txt
```

```bash
# 生成语料库
python generate_corpus.py

# 查看统计
python show_syntax_stats.py
```

生成结果在 `output/generated_corpus_*.json`

---

## 核心功能

- **模板泛化**: 1 个模板 → 数百个查询变体
- **智能控制**: 自动控制组合爆炸，避免生成过多查询
- **中文描述**: 自动生成流畅的查询描述
- **语法分析**: 统计生成查询的语法分布

---

## 项目结构

```
├── generate_corpus.py              # 主程序
├── gremlin_templates.csv           # 模板文件
├── config.json                     # 配置
├── base/
│   ├── generator.py                # 生成器
│   ├── Schema.py                   # Schema 管理
│   ├── GremlinBase.py              # 翻译引擎
│   ├── TraversalGenerator.py       # 遍历生成器
│   ├── combination_control_config.json  # 组合控制配置
│   └── gremlin/                    # ANTLR 解析器
├── db_data/                        # 数据和 Schema
└── output/                         # 输出目录
```

---

## 使用方式

### 1. 命令行

```bash
python generate_corpus.py
```

### 2. Python API

```python
from base import generate_gremlin_corpus

result = generate_gremlin_corpus(
    templates='gremlin_templates.csv',
    config_path='config.json',
    schema_path='db_data/schema/movie_schema.json',
    data_path='db_data/'
)

print(f"生成了 {result['total_unique_queries']} 个查询")
```

### 3. 添加模板

```bash
python add_template.py
```

或直接编辑 `gremlin_templates.csv`

---

## 配置说明

### 模板文件 (`gremlin_templates.csv`)

| 列名 | 说明 | 示例 |
|------|------|------|
| template | Gremlin 查询模板 | `g.V().hasLabel('person')` |
| description | 模板描述（可选） | 查询所有人 |

### 组合控制 (`base/combination_control_config.json`)

控制查询生成数量，详见 `COMBINATION_CONTROL_GUIDE.md`

核心参数：
- **链长度分类**: 短链(≤4步)、中链(5-6步)、长链(7-8步)、超长链(≥9步)
- **数据值填充**: 中间步骤填1个值，终端步骤填2-3个值
- **属性泛化**: 根据链长度动态调整泛化程度
- **查询数量限制**: 中链≤100，长链≤500，超长链≤50

---

## 输出格式

```json
{
  "metadata": {
    "total_templates": 198,
    "successful_templates": 198,
    "total_unique_queries": 1493,
    "generation_timestamp": "2025-10-29 19:07:33"
  },
  "corpus": [
    {
      "query": "g.V().hasLabel('person').out('acted_in')",
      "description": "从图中开始查找所有顶点，过滤出'人'类型的顶点，沿'参演'边out方向遍历"
    }
  ]
}
```

---

## 语法分析

生成语料库后，可以分析语法分布：

```bash
# 分析语法分布
python analyze_syntax_distribution.py

# 查看统计
python show_syntax_stats.py

# 可视化
python visualize_syntax_distribution.py
```

分析结果：
- `output/syntax_distribution_stats.json` - 统计数据
- `output/SYNTAX_ANALYSIS_SUMMARY.md` - 分析报告

---

## 工具脚本

| 脚本 | 功能 |
|------|------|
| `generate_corpus.py` | 生成语料库 |
| `add_template.py` | 添加新模板 |
| `list_templates.py` | 列出所有模板 |
| `analyze_syntax_distribution.py` | 分析语法分布 |
| `show_syntax_stats.py` | 查看统计摘要 |
| `visualize_syntax_distribution.py` | 可视化统计 |
| `verify_all.py` | 验证所有查询 |

---

## 文档导航

### 快速入门
- `QUICK_START.md` - 5分钟快速上手
- `QUICK_REFERENCE.md` - 常用命令速查

### 详细文档
- `TEMPLATES_GUIDE.md` - 模板编写指南
- `COMBINATION_CONTROL_GUIDE.md` - 组合控制配置说明
- `DEDUPLICATION_MECHANISM.md` - 去重机制说明

### 分析报告
- `SYNTAX_ANALYSIS_COMPLETE.md` - 语法分析完整报告
- `output/SYNTAX_ANALYSIS_SUMMARY.md` - 语法分析摘要
- `SYNTAX_QUICK_REFERENCE.md` - 语法统计速查

---

## 核心特性

### 1. 模板泛化
从一个模板生成多个变体：
```
模板: g.V().hasLabel('person').out('acted_in')

泛化:
→ g.V().hasLabel('movie').out('acted_in')
→ g.V().hasLabel('person').out('directed')
→ g.V().hasLabel('genre').out('has_genre')
...
```

### 2. 智能控制
- **链长度自适应**: 短链多泛化，长链少泛化
- **位置敏感**: 中间步骤保守，终端步骤充分
- **类型区分**: Schema 属性积极泛化，数据值保守填充

### 3. 自动去重
- 查询级去重（完全相同的查询）
- 语义级去重（等价查询）
- 保证生成的查询都是唯一的

### 4. 中文翻译
自动生成流畅的中文描述：
```
g.V().hasLabel('person').out('acted_in').has('title', 'Inception')
↓
从图中开始查找所有顶点，过滤出'人'类型的顶点，沿'参演'边out方向遍历，其'标题'为'Inception'
```

---

## 支持的 Gremlin 语法

### 起始步骤
`V()`, `E()`, `addV()`, `addE()`

### 导航步骤
`out()`, `in()`, `both()`, `outE()`, `inE()`, `bothE()`, `outV()`, `inV()`, `otherV()`

### 过滤步骤
`has()`, `hasLabel()`, `hasId()`, `hasKey()`, `hasValue()`, `where()`, `is()`, `not()`, `filter()`, `dedup()`, `simplePath()`, `cyclicPath()`

### 属性访问
`values()`, `properties()`, `valueMap()`, `elementMap()`, `label()`, `id()`

### 转换步骤
`order()`, `by()`, `limit()`, `range()`, `skip()`, `tail()`, `sample()`, `coin()`

### 聚合步骤
`count()`, `sum()`, `mean()`, `min()`, `max()`, `fold()`, `unfold()`, `group()`, `groupCount()`

### 分支步骤
`union()`, `coalesce()`, `choose()`, `optional()`

### 循环步骤
`repeat()`, `times()`, `until()`, `emit()`

### 路径步骤
`path()`, `tree()`, `as()`, `select()`

### 副作用步骤
`aggregate()`, `store()`, `sideEffect()`, `cap()`

### 谓词
`P.eq()`, `P.neq()`, `P.gt()`, `P.gte()`, `P.lt()`, `P.lte()`, `P.between()`, `P.inside()`, `P.outside()`, `P.within()`, `P.without()`

`TextP.startingWith()`, `TextP.endingWith()`, `TextP.containing()`, `TextP.notStartingWith()`, `TextP.notEndingWith()`, `TextP.notContaining()`, `TextP.regex()`, `TextP.notRegex()`

### 终端步骤
`next()`, `hasNext()`, `toList()`, `toSet()`, `iterate()`, `tryNext()`, `explain()`, `profile()`

---

## 统计数据（基于当前语料库）

- **总查询数**: 1,493
- **总步骤数**: 7,353
- **不同步骤类型**: 76 种
- **平均步骤/查询**: 4.92

### Top 10 步骤
1. `hasLabel` - 1,485 次 (20.20%)
2. `V` - 1,482 次 (20.16%)
3. `out` - 1,202 次 (16.35%)
4. `in` - 475 次 (6.46%)
5. `dedup` - 302 次 (4.11%)
6. `by` - 259 次 (3.52%)
7. `as` - 254 次 (3.45%)
8. `has` - 209 次 (2.84%)
9. `groupCount` - 182 次 (2.48%)
10. `where` - 147 次 (2.00%)

详见 `SYNTAX_QUICK_REFERENCE.md`

---

## 常见问题

### 生成的查询太多？
调整 `base/combination_control_config.json` 中的 `max_total_combinations`

### 生成的查询太少？
增加 `property_generalization` 中的 `additional_random_max`

### 某些模板失败？
运行 `python debug_failed_templates.py` 查看详细错误

### 如何验证生成的查询？
运行 `python verify_all.py` 验证所有查询的语法正确性

---


