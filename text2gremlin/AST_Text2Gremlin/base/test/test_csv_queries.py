#!/usr/bin/env python3
"""
测试CSV数据中的Gremlin查询脚本

从cypher2gremlin_dataset_thread.csv中读取gremlin_query，
尝试进行泛化生成，记录所有处理过程中的错误。
"""

import os
import sys
import csv
import json
import traceback
from datetime import datetime
from antlr4 import InputStream, CommonTokenStream
from antlr4.error.ErrorListener import ErrorListener

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Config import Config
from Schema import Schema
from GremlinBase import GremlinBase
from GremlinParse import Traversal
from TraversalGenerator import TraversalGenerator
from GremlinTransVisitor import GremlinTransVisitor

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

def check_gremlin_syntax(query_string: str) -> tuple[bool, str]:
    """
    检查给定的Gremlin查询语句的语法。
    
    Args:
        query_string: The Gremlin query to check.
        
    Returns:
        A tuple containing:
        - bool: True if syntax is correct, False otherwise.
        - str: An error message if syntax is incorrect, or "Syntax OK" if correct.
    """
    try:
        input_stream = InputStream(query_string)
        lexer = GremlinLexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        parser = GremlinParser(token_stream)
        
        # 移除默认的控制台错误监听器
        parser.removeErrorListeners()
        
        # 添加自定义的监听器
        error_listener = SyntaxErrorListener()
        parser.addErrorListener(error_listener)
        
        # 尝试解析查询
        parser.queryList()
        
        if error_listener.has_error:
            return (False, error_listener.error_message)
        else:
            return (True, "Syntax OK")
            
    except Exception as e:
        return (False, f"Parser Exception: {str(e)}")

def test_single_query(query: str, config: Config, schema: Schema, gremlin_base: GremlinBase) -> tuple[bool, str]:
    """
    测试单个Gremlin查询的泛化生成过程
    
    Args:
        query: Gremlin查询字符串
        config: 配置对象
        schema: Schema对象
        gremlin_base: 翻译对象
        
    Returns:
        tuple[bool, str]: (是否成功, 错误信息)
    """
    try:
        # 步骤1: 语法检查
        is_valid, syntax_error = check_gremlin_syntax(query)
        if not is_valid:
            return False, f"Syntax Error: {syntax_error}"
        
        # 步骤2: ANTLR解析为AST并提取配方
        visitor = GremlinTransVisitor()
        recipe = visitor.parse_and_visit(query)
        
        if not recipe or not recipe.steps:
            return False, "Recipe extraction failed: Empty recipe"
        
        # 步骤3: 尝试泛化生成
        generator = TraversalGenerator(schema, recipe, gremlin_base)
        corpus = generator.generate()
        
        if not corpus:
            return False, "Generation failed: No queries generated"
        
        # 步骤4: 检查生成的查询语法
        syntax_errors = []
        for generated_query, _ in corpus:
            is_gen_valid, gen_error = check_gremlin_syntax(generated_query)
            if not is_gen_valid:
                syntax_errors.append(f"Generated query syntax error: {gen_error}")
        
        if syntax_errors:
            return False, f"Generated queries have syntax errors: {'; '.join(syntax_errors[:3])}"  # 只显示前3个错误
        
        return True, f"Success: Generated {len(corpus)} queries"
        
    except Exception as e:
        # 捕获所有其他异常
        error_trace = traceback.format_exc()
        return False, f"Exception: {str(e)}\nTrace: {error_trace}"

def load_csv_queries(csv_file_path: str) -> list[tuple[str, str]]:
    """
    从CSV文件中加载查询数据
    
    Args:
        csv_file_path: CSV文件路径
        
    Returns:
        list[tuple[str, str]]: (question, gremlin_query)的列表
    """
    queries = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                question = row.get('question', '').strip()
                gremlin_query = row.get('gremlin_query', '').strip()
                if gremlin_query:  # 只处理非空的查询
                    queries.append((question, gremlin_query))
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return []
    
    return queries

def save_error_results(error_results: list[tuple[str, str]], output_file: str):
    """
    保存错误结果到CSV文件
    
    Args:
        error_results: (gremlin_query, error_message)的列表
        output_file: 输出文件路径
    """
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['gremlin_query', 'error_message'])  # 写入表头
            for query, error in error_results:
                writer.writerow([query, error])
        print(f"Error results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """主函数"""
    print("=== Gremlin查询测试脚本 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 配置文件路径
    csv_file_path = "cypher2gremlin_dataset_thread.csv"
    
    # 获取项目根目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_root = os.path.dirname(current_dir)  
    project_root = os.path.dirname(base_root)
    config_path = os.path.join(project_root, 'config.json')
    schema_path = os.path.join(project_root, 'db_data', 'schema', 'movie_schema.json')
    data_path = os.path.join(project_root, 'db_data')
    
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误: CSV文件不存在: {csv_file_path}")
        return
    
    # 加载配置和数据
    print("加载配置和数据...")
    try:
        config = Config(config_path)
        schema = Schema(schema_path, data_path)
        gremlin_base = GremlinBase(config)
        print("配置和数据加载成功")
    except Exception as e:
        print(f"加载配置失败: {e}")
        return
    
    # 加载CSV查询数据
    print(f"从 {csv_file_path} 加载查询数据...")
    queries = load_csv_queries(csv_file_path)
    print(f"加载了 {len(queries)} 条查询")
    
    if not queries:
        print("没有找到有效的查询数据")
        return
    
    # 测试每个查询
    print("开始测试查询...")
    error_results = []
    success_count = 0
    total_count = len(queries)
    
    for i, (question, gremlin_query) in enumerate(queries, 1):
        print(f"\n进度: {i}/{total_count} ({i/total_count*100:.1f}%)")
        print(f"测试查询: {gremlin_query[:100]}{'...' if len(gremlin_query) > 100 else ''}")
        
        success, error_message = test_single_query(gremlin_query, config, schema, gremlin_base)
        
        if success:
            success_count += 1
            print(f"✅ 成功: {error_message}")
        else:
            error_results.append((gremlin_query, error_message))
            print(f"❌ 失败: {error_message[:200]}{'...' if len(error_message) > 200 else ''}")
    
    # 生成结果报告
    print(f"\n=== 测试完成 ===")
    print(f"总查询数: {total_count}")
    print(f"成功处理: {success_count} ({success_count/total_count*100:.1f}%)")
    print(f"处理失败: {len(error_results)} ({len(error_results)/total_count*100:.1f}%)")
    
    # 保存错误结果
    if error_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        error_output_file = f"gremlin_query_errors_{timestamp}.csv"
        save_error_results(error_results, error_output_file)
        
        # 显示错误类型统计
        print(f"\n=== 错误类型统计 ===")
        error_types = {}
        for _, error_msg in error_results:
            error_type = error_msg.split(':')[0] if ':' in error_msg else 'Unknown'
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"{error_type}: {count} ({count/len(error_results)*100:.1f}%)")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()