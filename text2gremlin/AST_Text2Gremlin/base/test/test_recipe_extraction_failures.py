#!/usr/bin/env python3
"""
测试Recipe extraction failed
"""

import os
import sys
import csv
import traceback
from datetime import datetime


from antlr4 import InputStream, CommonTokenStream
from antlr4.error.ErrorListener import ErrorListener

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GremlinTransVisitor import GremlinTransVisitor
from GremlinParse import Traversal


from gremlin.GremlinLexer import GremlinLexer
from gremlin.GremlinParser import GremlinParser

class SyntaxErrorListener(ErrorListener):
    """私有错误监听器类，捕获语法错误。"""
    
    def __init__(self):
        super().__init__()
        self.has_error = False
        self.error_message = ""
    
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        """当语法错误发生时，此方法被调用。"""
        self.has_error = True
        self.error_message = f"Syntax Error at line {line}, column {column}: {msg}"

def extract_recipe_extraction_failures(csv_file_path: str) -> list[str]:
    """
    从CSV文件中提取Recipe extraction failed的查询
    
    Args:
        csv_file_path: CSV文件路径
        
    Returns:
        list[str]: 失败的查询列表
    """
    failed_queries = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                error_message = row.get('error_message', '').strip()
                if 'Recipe extraction failed: Empty recipe' in error_message:
                    gremlin_query = row.get('gremlin_query', '').strip()
                    if gremlin_query:
                        failed_queries.append(gremlin_query)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return []
    
    return failed_queries

def test_recipe_extraction(query: str) -> tuple[bool, str, object]:
    """
    测试单个查询的Recipe提取过程
    
    Args:
        query: Gremlin查询字符串
        
    Returns:
        tuple[bool, str, object]: (是否成功, 错误信息, Recipe对象)
    """
    try:
        # 步骤1: 语法检查
        input_stream = InputStream(query)
        lexer = GremlinLexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        parser = GremlinParser(token_stream)
        
        # 移除默认的控制台错误监听器
        parser.removeErrorListeners()
        
        # 添加自定义的监听器
        error_listener = SyntaxErrorListener()
        parser.addErrorListener(error_listener)
        
        # 尝试解析查询
        tree = parser.queryList()
        
        if error_listener.has_error:
            return False, f"Syntax Error: {error_listener.error_message}", None
        
        # 步骤2: 尝试提取Recipe
        visitor = GremlinTransVisitor()
        recipe = visitor.visit(tree)
        
        if not recipe:
            return False, "Recipe is None", None
        
        if not hasattr(recipe, 'steps') or not recipe.steps:
            return False, f"Recipe has no steps: {type(recipe)}, {recipe}", recipe
        
        return True, f"Success: Recipe has {len(recipe.steps)} steps", recipe
        
    except Exception as e:
        # 捕获所有其他异常
        error_trace = traceback.format_exc()
        return False, f"Exception: {str(e)}\nTrace: {error_trace}", None

def analyze_query_structure(query: str):
    """
    分析查询的结构，帮助理解为什么提取失败
    """
    print(f"\n=== 查询结构分析 ===")
    print(f"查询长度: {len(query)}")
    print(f"是否包含g.call: {'g.call' in query}")
    print(f"是否包含g.V(): {'g.V()' in query}")
    print(f"是否包含g.E(): {'g.E()' in query}")
    print(f"是否包含g.inject: {'g.inject' in query}")
    
    # 分析主要的步骤类型
    steps = []
    if 'g.call(' in query:
        steps.append('call')
    if 'g.V(' in query:
        steps.append('V')
    if 'g.E(' in query:
        steps.append('E')
    if 'g.inject(' in query:
        steps.append('inject')
    
    print(f"检测到的起始步骤: {steps}")
    
    # 检查是否有复杂的嵌套结构
    if '.with(' in query:
        print("包含.with()调用")
    if '.select(' in query:
        print("包含.select()调用")
    if '.project(' in query:
        print("包含.project()调用")

def main():
    """主函数"""
    print("=== Recipe Extraction 失败分析脚本 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 配置文件路径
    csv_file_path = "gremlin_query_errors_20250928_030206.csv"
    
    if not os.path.exists(csv_file_path):
        print(f"错误: CSV文件不存在: {csv_file_path}")
        return
    
    # 提取失败的查询
    print(f"从 {csv_file_path} 提取Recipe extraction failed的查询...")
    failed_queries = extract_recipe_extraction_failures(csv_file_path)
    print(f"找到 {len(failed_queries)} 条失败的查询")
    
    if not failed_queries:
        print("没有找到Recipe extraction failed的查询")
        return
    
    # 分析每个失败的查询
    print("\n开始详细分析每个失败的查询...")
    success_count = 0
    total_count = len(failed_queries)
    detailed_results = []
    
    for i, query in enumerate(failed_queries, 1):
        print(f"\n{'='*80}")
        print(f"进度: {i}/{total_count} ({i/total_count*100:.1f}%)")
        print(f"查询: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        # 分析查询结构
        analyze_query_structure(query)
        
        # 测试Recipe提取
        success, error_message, recipe = test_recipe_extraction(query)
        
        if success:
            success_count += 1
            print(f"✅ 成功: {error_message}")
            if recipe and hasattr(recipe, 'steps'):
                print(f"   Recipe步骤: {[step.name for step in recipe.steps]}")
        else:
            print(f"❌ 失败: {error_message}")
            detailed_results.append({
                'query': query,
                'error': error_message,
                'query_type': 'call' if 'g.call(' in query else ('E' if 'g.E(' in query else 'other')
            })
    
    # 生成分析报告
    print(f"\n{'='*80}")
    print(f"=== 分析完成 ===")
    print(f"总查询数: {total_count}")
    print(f"成功提取: {success_count} ({success_count/total_count*100:.1f}%)")
    print(f"仍然失败: {len(detailed_results)} ({len(detailed_results)/total_count*100:.1f}%)")
    
    # 按错误类型分组
    if detailed_results:
        print(f"\n=== 失败查询类型分析 ===")
        error_types = {}
        query_types = {}
        
        for result in detailed_results:
            # 错误类型统计
            error_type = result['error'].split(':')[0] if ':' in result['error'] else 'Unknown'
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # 查询类型统计
            query_type = result['query_type']
            query_types[query_type] = query_types.get(query_type, 0) + 1
        
        print("错误类型分布:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count} ({count/len(detailed_results)*100:.1f}%)")
        
        print("查询类型分布:")
        for query_type, count in sorted(query_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {query_type}: {count} ({count/len(detailed_results)*100:.1f}%)")
        
        # 保存详细的失败结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        detailed_output_file = f"./test/recipe_extraction_detailed_failures_{timestamp}.csv"
        
        try:
            with open(detailed_output_file, 'w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['query', 'error_message', 'query_type'])
                for result in detailed_results:
                    writer.writerow([result['query'], result['error'], result['query_type']])
            print(f"\n详细失败结果保存到: {detailed_output_file}")
        except Exception as e:
            print(f"保存详细结果失败: {e}")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()