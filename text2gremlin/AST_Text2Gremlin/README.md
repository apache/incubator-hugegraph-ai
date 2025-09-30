# Gremlin语料库生成系统 (base_v2)

## 📖 项目背景

本项目是一个基于ANTLR的Gremlin查询语料库生成系统，专为Text-to-Gremlin任务提供高质量的训练数据。系统能够从少量的Gremlin查询模板出发，通过智能泛化和随机增强技术，生成大量语法正确、语义丰富的Gremlin查询及其对应的中文自然语言描述。

### 🎯 核心价值

- **数据稀缺问题解决**: 解决Text-to-Gremlin任务中训练数据不足的问题
- **高质量语料生成**: 生成语法正确、语义准确的查询-描述对
- **智能泛化能力**: 从少量模板生成大量多样化的查询变体
- **中文本土化**: 提供准确的中文自然语言描述

## 🔧 核心原理

### 1. Gremlin泛化原理

系统采用**基于Schema的智能泛化**策略：

```
模板查询 → ANTLR解析 → 结构化Recipe → Schema引导泛化 → 多样化查询
```

**泛化维度：**
- **实体泛化**: `person` → `movie`, `user`, `genre`等
- **属性泛化**: `name` → `title`, `born`, `duration`等  
- **关系泛化**: `acted_in` → `directed`, `produce`, `rate`等
- **数据泛化**: 从CSV文件中获取真实数据值进行替换

### 2. 翻译原理

采用**多层次翻译机制**：

```
英文术语 → Schema字典查询 → 中文术语 → 模板格式化 → 自然语言描述
```

**翻译层次：**
- **Schema术语翻译**: `movie`→`电影`, `title`→`标题`
- **操作步骤翻译**: `.out()`→`沿着...的出边`, `.has()`→`其...属性为`
- **上下文感知**: 根据查询上下文选择合适的翻译

### 3. 随机增强原理

基于**查询状态感知的随机增强**：

```
查询状态判断 → 增强类型选择 → 概率触发 → 语义保持增强
```

**增强策略：**
- **元素流增强**: `limit()`, `sample()`, `range()`, `dedup()`, `order()`
- **值流增强**: 针对`.values()`后的数据流进行优化增强
- **智能避免**: 自动识别终止步骤，避免无效增强

## 📁 项目结构

```
base/
├── README.md                 # 项目说明文档
├── generator.py              # 🚀 主入口脚本
├── Config.py                 # 配置管理模块
├── Schema.py                 # 📊 Schema和数据管理
├── GremlinBase.py           # 🌐 翻译引擎
├── GremlinParse.py          # 📝 数据结构定义
├── TraversalGenerator.py    # ⚡ 核心生成引擎
├── GremlinTransVisitor.py   # 🔄 AST访问器
├── template/                # 📚 翻译字典
│   ├── schema_dict.txt      # Schema术语翻译
│   └── syn_dict.txt         # 同义词字典
└── gremlin/                 # 🔧 ANTLR生成的解析器
    ├── GremlinLexer.py
    ├── GremlinParser.py
    └── ...
```
```
AST_Text2Gremlin/                   # 项目根目录
├── base/                           # 核心系统目录
│   ├── generator.py                # 主生成器入口
│   ├── GremlinTransVisitor.py      # ANTLR语法树访问器
│   ├── TraversalGenerator.py       # 递归回溯生成器
│   ├── Schema.py                   # 图数据库Schema管理
│   ├── GremlinBase.py              # 基础组件库
│   ├── Config.py                   # 配置管理
│   ├── cypher2gremlin_dataset.csv  # 3514条真实查询数据集
│   └── test/                       # 测试套件
├── config.json                     # 全局配置文件
├── db_data/                        # Schema和数据文件
└── README.md                       # 详细技术文档

这样就把所有文件都包含在 AST_Text2Gremlin 项目目录下了。
```
## 🧩 核心模块详解

### 🚀 generator.py - 主入口脚本
**作用**: 系统的主要入口点，协调整个生成流程
**核心功能**:
- 模板批处理和全局去重
- 语法检查和质量保证
- 统计分析和结果展示
- 支持函数化调用接口

**使用方式**:
```python
# 直接运行
python generator.py

# 函数调用
from generator import generate_corpus_from_templates
result = generate_corpus_from_templates(templates)
```

### 📊 Schema.py - Schema和数据管理
**作用**: 管理图数据库的Schema信息和实例数据
**核心功能**:
- 加载和解析JSON格式的Schema定义
- 从CSV文件中读取真实数据实例
- 提供Schema查询接口（标签、属性、关系等）
- 支持多实例数据获取，增加泛化多样性

**关键方法**:
```python
schema.get_vertex_labels()           # 获取所有顶点标签
schema.get_properties_with_type()    # 获取属性及类型信息
schema.get_instances()               # 获取多个数据实例
schema.get_valid_steps()             # 获取有效的遍历步骤
```

### ⚡ TraversalGenerator.py - 核心生成引擎
**作用**: 系统的核心，负责从Recipe生成大量查询变体
**核心功能**:
- 递归深度优先搜索生成所有可能的查询路径
- 基于Schema的智能泛化（实体、属性、关系、数据）
- 随机增强功能（20%末端 + 10%中间概率触发）
- 中间结果保存，生成不同复杂度的查询

**生成策略**:
```python
# 泛化维度
- 标签泛化: person → movie, user, genre
- 属性泛化: name → title, born, duration  
- 关系泛化: acted_in → directed, produce
- 数据泛化: 真实CSV数据替换

# 随机增强
- 元素流: limit(), sample(), range(), dedup(), order()
- 值流: 针对values()优化的增强
- 智能判断: 避免在终止步骤后增强
```

### 🔄 GremlinTransVisitor.py - AST访问器
**作用**: 将ANTLR解析的AST转换为结构化的Recipe对象
**核心功能**:
- 实现ANTLR Visitor模式
- 处理各种Gremlin语法结构
- 提取查询的核心步骤和参数
- 生成可用于泛化的Recipe

**支持的语法**:
- 遍历步骤: `.V()`, `.out()`, `.in()`, `.both()`
- 过滤步骤: `.has()`, `.hasLabel()`, `.limit()`
- 修改步骤: `.addV()`, `.property()`, `.drop()`
- 复杂结构: 嵌套遍历、谓词表达式等

### 🌐 GremlinBase.py - 翻译引擎
**作用**: 提供Gremlin术语到中文的智能翻译
**核心功能**:
- 加载Schema翻译字典和同义词字典
- 提供上下文感知的术语翻译
- 支持随机同义词选择，增加描述多样性
- 格式化生成自然流畅的中文描述

**翻译机制**:
```python
# 术语翻译
movie → 电影
title → 标题  
acted_in → 参演

# 操作翻译
.out('acted_in') → "沿着'参演'的出边"
.has('title', 'Matrix') → "其'标题'属性为'Matrix'"
.limit(10) → "并只取前10个结果"
```

### 📝 其他支持模块

**Config.py**: 配置文件管理，支持JSON格式配置
**GremlinParse.py**: 定义Step和Traversal数据结构
**template/**: 翻译字典文件，支持Schema术语和同义词翻译

## 🎯 使用示例

### 基本使用
```python
# 定义模板
templates = [
    "g.V().has('name', 'John')",
    "g.V().hasLabel('movie').out('acted_in')", 
    "g.addV('person').property('name', 'Test')",
    "g.V().has('title', 'Matrix').drop()"
]

# 生成语料库
from generator import generate_corpus_from_templates
result = generate_corpus_from_templates(templates)

print(f"生成了 {result['total_unique_queries']} 个独特查询")
```

### 生成结果示例
```json
{
  "query": "g.V().hasLabel('person').out('acted_in').limit(10)",
  "description": "从图中开始查找所有顶点，过滤出'人'类型的顶点，然后沿着'参演'的出边找到'电影'顶点，并只取前10个结果"
}
```

## 📈 系统特性

### 🔥 高质量保证
- **语法检查**: 使用ANTLR确保100%语法正确
- **去重机制**: 全局去重避免重复查询
- **类型安全**: 基于Schema的类型检查

### ⚡ 高效生成
- **智能泛化**: 从32个模板生成940+独特查询
- **批量处理**: 支持大规模模板批处理
- **增量生成**: 支持增量添加新模板

### 🌟 丰富多样性
- **操作类型均衡**: 查询62.6% + 更新28.3% + 创建6.6% + 删除2.6%
- **复杂度分布**: 1-8步查询全覆盖
- **随机增强**: 20%末端 + 10%中间增强概率

### 🌐 本土化支持
- **中文翻译**: 准确的Schema术语翻译
- **自然描述**: 符合中文表达习惯的自然语言
- **可扩展字典**: 支持自定义翻译字典

## 🚀 快速开始

### 1. 环境准备
```bash
# 确保Python环境和依赖
pip install antlr4-python3-runtime
```

### 2. 运行生成
```bash
cd base_v2
python generator.py
```

### 3. 查看结果
生成的语料库保存在 `generated_corpus.json` 文件中，包含：
- 元数据信息（模板数量、查询数量等）
- 查询-描述对列表
- 完整的JSON格式，便于后续处理

## 🔧 自定义配置

### 添加新的Schema术语翻译
编辑 `template/schema_dict.txt`:
```
new_entity 新实体
new_property 新属性
new_relation 新关系
```

### 调整随机增强概率
修改 `TraversalGenerator.py` 中的概率参数：
```python
if random.random() < 0.2:  # 末端增强概率
if random.random() < 0.1:  # 中间增强概率
```

### 扩展支持的Gremlin语法
在 `GremlinTransVisitor.py` 中添加新的visit方法。

## 📊 性能指标

- **生成效率**: 32个模板 → 940个查询 (29.4倍放大)
- **语法正确率**: 100% (经ANTLR验证)
- **翻译准确率**: 基于人工标注的Schema字典
- **多样性指标**: 8种复杂度级别，4种操作类型均衡分布

## 🤝 贡献指南

欢迎贡献代码和改进建议！主要贡献方向：
- 扩展Gremlin语法支持
- 改进翻译质量和多样性
- 优化生成算法效率
- 添加新的随机增强策略

## 📄 许可证

本项目采用开源许可证，详见LICENSE文件。

---

**项目维护**: 本项目专注于为Text-to-Gremlin任务提供高质量的训练数据，持续改进和优化中。