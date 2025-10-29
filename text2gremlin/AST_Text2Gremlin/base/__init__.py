"""
Gremlin 查询生成器包

这个包提供了从模板生成 Gremlin 查询语料库的功能。

主要模块:
- generator: 主要的生成器接口
- Config: 配置管理
- Schema: 图数据库模式定义
- TraversalGenerator: 遍历查询生成器
- GremlinTransVisitor: Gremlin 语法解析器
"""

__version__ = "1.0.0"

# 导出主要接口
from .generator import generate_gremlin_corpus

__all__ = ['generate_gremlin_corpus']
