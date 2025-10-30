# Gremlin 查询语料库生成器

从模板生成大量多样化的 Gremlin 查询及其中文描述，用于测试、训练和分析。

## 快速开始
环境配置：python：3.12.10
```bash
pip install -r requirements.txt
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

```text
├── generate_corpus.py                   # 主程序
├── gremlin_templates.csv                # 模板文件
├── config.json                          # 配置
├── base/
│   ├── generator.py                     # 解析泛化控制器
│   ├── Config.py                        # 配置管理模块
│   ├── Schema.py                        # Schema和数据管理
│   ├── GremlinParse.py                  # 数据结构定义
│   ├── GremlinExpr.py                   # 复杂表达式定义(谓词、匿名遍历等)
│   ├── GremlinTransVisitor.py           # AST解析
│   ├── TraversalGenerator.py            # 遍历生成器
│   ├── combination_control_config.json  # 组合控制配置
│   ├── GremlinBase.py                   # 翻译引擎
│   ├── gremlin/                         # ANTLR生成的解析器
│   └── template/                        # 翻译字典
│       ├── schema_dict.txt              # Schema术语翻译
│       └── syn_dict.txt                 # 同义词字典
├── db_data/                             # 数据和 Schema
└── output/                              # 输出目录
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

直接编辑 `gremlin_templates.csv`即可

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

## 核心特性

### 1. 模板泛化
从一个模板生成多个变体：
```text
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
```text
g.V().hasLabel('person').out('acted_in').has('title', 'Inception')
↓
从图中开始查找所有顶点，过滤出'人'类型的顶点，沿'参演'边out方向遍历，其'标题'为'Inception'
```



