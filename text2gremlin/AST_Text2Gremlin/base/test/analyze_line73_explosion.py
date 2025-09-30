#!/usr/bin/env python3
"""
详细分析第73行查询的组合爆炸原因
计算每个步骤的可能组合数量
"""

import csv
import os
from GremlinTransVisitor import GremlinTransVisitor
from TraversalGenerator import TraversalGenerator
from Config import Config
from Schema import Schema
from GremlinBase import GremlinBase

def analyze_schema_matches():
    """分析Schema中的匹配情况"""
    print("🔍 分析Schema中的数据...")
    
    # 加载配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    schema_path = os.path.join(project_root, 'db_data', 'schema', 'movie_schema.json')
    data_path = os.path.join(project_root, 'db_data', 'movie')
    config_path = os.path.join(project_root, 'config.json')
    
    config = Config(file_path=config_path)
    schema = Schema(schema_path, data_path)
    
    print("\n📊 Schema统计信息:")
    
    # 获取标签信息
    vertex_labels = schema.get_vertex_labels()
    edge_labels = schema.get_edge_labels()
    
    print(f"顶点标签数量: {len(vertex_labels)}")
    print(f"顶点标签: {vertex_labels}")
    print(f"边标签数量: {len(edge_labels)}")
    print(f"边标签: {edge_labels}")
        
    return schema

def analyze_recipe_steps(query: str):
    """分析Recipe的步骤分解"""
    print(f"\n🔬 分析查询Recipe:")
    print(f"原查询: {query}")
    
    visitor = GremlinTransVisitor()
    recipe = visitor.parse_and_visit(query)
    
    if not recipe or not hasattr(recipe, 'steps'):
        print("❌ Recipe提取失败")
        return None
    
    print(f"\n📋 Recipe包含 {len(recipe.steps)} 个步骤:")
    for i, step in enumerate(recipe.steps, 1):
        print(f"  步骤{i}: {step.name}({step.params})")
        
    return recipe

def calculate_combinations(schema, recipe):
    """计算每个步骤的可能组合数"""
    print(f"\n🧮 计算组合数量:")
    
    # 获取schema中的标签和边
    vertex_labels = schema.get_vertex_labels()
    edge_labels = schema.get_edge_labels()
    
    print(f"Schema信息:")
    print(f"  顶点标签: {vertex_labels} (共{len(vertex_labels)}个)")
    print(f"  边标签: {edge_labels} (共{len(edge_labels)}个)")
    
    # 分析每个可泛化的步骤
    step_combinations = []
    
    for i, step in enumerate(recipe.steps, 1):
        step_name = step.name
        step_params = step.params
        
        print(f"\n步骤{i}: {step_name}({step_params})")
        
        if step_name == 'hasLabel':
            # hasLabel步骤可以替换为其他标签
            possible_labels = len(vertex_labels)
            print(f"  可替换为任意顶点标签: {possible_labels} 种可能")
            step_combinations.append(possible_labels)
                
        elif step_name in ['out', 'in', 'both']:
            # 导航步骤可以替换为其他边标签
            possible_edges = len(edge_labels)
            print(f"  可替换为任意边标签: {possible_edges} 种可能")
            step_combinations.append(possible_edges)
                
        elif step_name in ['outE', 'inE', 'bothE']:
            # 边遍历步骤
            possible_edges = len(edge_labels)
            print(f"  可替换为任意边标签: {possible_edges} 种可能")
            step_combinations.append(possible_edges)
                
        else:
            # 其他步骤暂时不泛化
            print(f"  不泛化: 1种可能")
            step_combinations.append(1)
    
    return step_combinations

def estimate_total_combinations(step_combinations):
    """估算总组合数"""
    print(f"\n📈 组合数计算:")
    
    total = 1
    for i, count in enumerate(step_combinations, 1):
        print(f"  步骤{i}: {count} 种可能")
        total *= count
    
    print(f"\n🎯 理论最大组合数: {total:,}")
    
    # 考虑实际约束
    print(f"\n⚠️  实际约束因素:")
    print(f"  - Schema连接性约束: 不是所有标签组合都有效")
    print(f"  - 语法约束: 某些组合可能产生无效查询")
    print(f"  - 去重机制: 相同查询会被去重")
    
    # 估算实际生成数
    constraint_factor = 0.1  # 假设约束因子为10%
    estimated_actual = int(total * constraint_factor)
    print(f"  估算实际生成数: {estimated_actual:,} (约束因子: {constraint_factor*100}%)")
    
    return total, estimated_actual

def analyze_actual_generation(query: str, schema: Schema, config: Config, gremlin_base: GremlinBase):
    """分析实际生成过程"""
    print(f"\n🚀 实际生成分析:")
    
    visitor = GremlinTransVisitor()
    recipe = visitor.parse_and_visit(query)
    
    generator = TraversalGenerator(schema, recipe, gremlin_base)
    corpus = generator.generate()
    
    print(f"实际生成数量: {len(corpus):,}")
    
    # 分析生成的查询类型
    label_patterns = {}
    edge_patterns = {}
    
    for query_gen, desc in corpus[:100]:  # 只分析前100个
        # 统计标签模式
        if '.hasLabel(' in query_gen:
            labels = []
            parts = query_gen.split('.hasLabel(')
            for part in parts[1:]:
                if ')' in part:
                    label = part.split(')')[0].strip("'\"")
                    labels.append(label)
            label_pattern = '->'.join(labels)
            label_patterns[label_pattern] = label_patterns.get(label_pattern, 0) + 1
        
        # 统计边模式
        if '.out(' in query_gen:
            edges = []
            parts = query_gen.split('.out(')
            for part in parts[1:]:
                if ')' in part:
                    edge = part.split(')')[0].strip("'\"")
                    edges.append(edge)
            edge_pattern = '->'.join(edges)
            edge_patterns[edge_pattern] = edge_patterns.get(edge_pattern, 0) + 1
    
    print(f"\n📊 生成模式分析 (前100个查询):")
    print(f"标签模式数量: {len(label_patterns)}")
    print(f"边模式数量: {len(edge_patterns)}")
    
    # 显示最常见的模式
    if label_patterns:
        print(f"\n最常见的标签模式:")
        for pattern, count in sorted(label_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {pattern}: {count} 次")
    
    if edge_patterns:
        print(f"\n最常见的边模式:")
        for pattern, count in sorted(edge_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {pattern}: {count} 次")

def main():
    """主函数"""
    print("🎯 详细分析第73行查询的组合爆炸")
    print("="*80)
    
    # 读取第73行查询
    with open('cypher2gremlin_dataset.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    line_73_query = rows[72]['gremlin_query']  # 第73行，索引为72
    if line_73_query.startswith('"') and line_73_query.endswith('"'):
        line_73_query = line_73_query[1:-1]
    
    print(f"第73行查询: {line_73_query}")
    
    # 1. 分析Schema
    schema = analyze_schema_matches()
    
    # 2. 分析Recipe步骤
    recipe = analyze_recipe_steps(line_73_query)
    
    if recipe:
        # 3. 计算理论组合数
        step_combinations = calculate_combinations(schema, recipe)
        total_combinations, estimated_actual = estimate_total_combinations(step_combinations)
        
        # 4. 分析实际生成
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, 'config.json')
        config = Config(file_path=config_path)
        gremlin_base = GremlinBase(config)
        
        analyze_actual_generation(line_73_query, schema, config, gremlin_base)
        
        print(f"\n" + "="*80)
        print(f"📋 总结:")
        print(f"  理论最大组合: {total_combinations:,}")
        print(f"  估算实际生成: {estimated_actual:,}")
        print(f"  实际观察到的: 6,908")
        print(f"  约束效率: {6908/total_combinations*100:.2f}%")
        print("="*80)

if __name__ == '__main__':
    main()