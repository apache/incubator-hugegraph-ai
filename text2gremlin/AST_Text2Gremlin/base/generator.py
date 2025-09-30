
"""
Gremlin语料库生成器主入口脚本。

从Gremlin查询模板生成大量多样化的查询-描述对，用于Text-to-Gremlin任务的训练数据。
"""

import os
import json
from antlr4 import InputStream, CommonTokenStream
from antlr4.error.ErrorListener import ErrorListener

# Import all our custom modules from the gremlin_base package
from Config import Config
from Schema import Schema
from GremlinBase import GremlinBase
from GremlinParse import Traversal
from TraversalGenerator import TraversalGenerator
from GremlinTransVisitor import GremlinTransVisitor

# Import the ANTLR-generated components
from gremlin.GremlinLexer import GremlinLexer
from gremlin.GremlinParser import GremlinParser
import random

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

def generate_corpus_from_template(
    template_string: str,
    config: Config,
    schema: Schema,
    gremlin_base: GremlinBase,
    global_corpus_dict: dict
) -> tuple[int, dict]:
    """
    执行单个 Gremlin 模板字符串的完整 pipeline。

    Args:
        template_string: 用作模板的 Gremlin query。
        config: 加载的 Config 对象。
        schema: 加载的 Schema 对象。
        gremlin_base: 加载的 GremlinBase 对象。
        global_corpus_dict: 用于存储唯一 query-description 对的全局字典。

    Returns:
        tuple: (添加到全局语料库的新的唯一对的数量, 处理统计信息)
    """
    # 初始化统计信息
    stats = {
        'success': False,
        'error_stage': '',
        'error_message': '',
        'generated_count': 0,
        'new_pairs_count': 0,
        'duplicate_count': 0,
        'syntax_error_count': 0
    }

    try:
        # ANTLR 解析为 AST,并提取模版
        visitor = GremlinTransVisitor()
        recipe = visitor.parse_and_visit(template_string)
        
        if not recipe:
            stats['error_stage'] = 'recipe_extraction'
            stats['error_message'] = 'Recipe extraction failed'
            return 0, stats
        
        if not hasattr(recipe, 'steps') or not recipe.steps:
            stats['error_stage'] = 'recipe_validation'
            stats['error_message'] = 'Recipe has no steps'
            return 0, stats

        # 泛化
        generator = TraversalGenerator(schema, recipe, gremlin_base)
        corpus = generator.generate()
        
        if not corpus:
            stats['error_stage'] = 'generation'
            stats['error_message'] = 'Generator returned empty corpus'
            return 0, stats
        
        stats['generated_count'] = len(corpus)
        
        # 语法检查 & 全局去重
        new_pairs_count = 0
        duplicate_count = 0
        syntax_error_count = 0
        
        for query, description in corpus:
            try:
                # 首先进行语法检查
                is_valid, error_msg = check_gremlin_syntax(query)
                
                if not is_valid:
                    syntax_error_count += 1
                    continue
                    
                if query not in global_corpus_dict:
                    # 新的查询且语法正确，添加到全局字典
                    global_corpus_dict[query] = description
                    new_pairs_count += 1
                else:
                    # 重复的查询，跳过
                    duplicate_count += 1
                    
            except Exception as e:
                syntax_error_count += 1
                continue
        
        # 更新统计信息
        stats['new_pairs_count'] = new_pairs_count
        stats['duplicate_count'] = duplicate_count
        stats['syntax_error_count'] = syntax_error_count
        stats['success'] = True
        
        # 添加生成数量的警告信息
        if stats['generated_count'] > 5000:
            stats['warning'] = f'由于本条模版的Recip复杂,生成了大量查询({stats["generated_count"]}条)'
        elif new_pairs_count == 0 and stats['generated_count'] > 0:
            stats['warning'] = f'生成了{stats["generated_count"]}条查询但全部重复'
        
        return new_pairs_count, stats
        
    except Exception as e:
        # 捕获所有其他异常
        stats['error_stage'] = 'unknown'
        stats['error_message'] = str(e)
        return 0, stats


def generate_corpus_from_templates(templates: list[str], 
                                  config_path: str = None, 
                                  schema_path: str = None, 
                                  data_path: str = None,
                                  output_file: str = "generated_corpus.json") -> dict:
    """
    从Gremlin模板列表生成完整的语料库。
    
    Args:
        templates: Gremlin查询模板列表
        config_path: 配置文件路径
        schema_path: Schema文件路径  
        data_path: 数据文件路径
        output_file: 输出文件名
        
    Returns:
        包含生成统计信息的字典
    """
    # --- Setup: Define paths and load dependencies ---
    if not config_path or not schema_path or not data_path:
        # 自动检测项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # 从base目录向上一级
        
        schema_path = schema_path or os.path.join(project_root, 'db_data', 'schema', 'movie_schema.json')
        data_path = data_path or os.path.join(project_root, 'db_data')
        config_path = config_path or os.path.join(project_root, 'config.json')

    if not all(os.path.exists(p) for p in [config_path, schema_path, data_path]):
        raise FileNotFoundError("Could not find necessary config, schema, or data files.")
        
    # Load all necessary components once
    config = Config(file_path=config_path)
    schema = Schema(schema_path, data_path)
    gremlin_base = GremlinBase(config)

    # --- Run the generation process for each template with global deduplication ---
    global_corpus_dict = {}  # 使用字典进行去重，key是query，value是description
    total_new_pairs = 0
    
    # 处理统计信息
    processing_stats = {
        'total_templates': len(templates),
        'successful_templates': 0,
        'failed_templates': 0,
        'failed_details': [],
        'total_generated': 0,
        'total_syntax_errors': 0,
        'total_duplicates': 0
    }
    
    print(f"🚀 开始处理 {len(templates)} 个模板...")
    
    for i, template in enumerate(templates, 1):
        try:
            new_pairs_count, template_stats = generate_corpus_from_template(
                template_string=template,
                config=config,
                schema=schema,
                gremlin_base=gremlin_base,
                global_corpus_dict=global_corpus_dict
            )
            
            total_new_pairs += new_pairs_count
            
            # 更新统计信息
            if template_stats['success']:
                processing_stats['successful_templates'] += 1
                processing_stats['total_generated'] += template_stats['generated_count']
                processing_stats['total_syntax_errors'] += template_stats['syntax_error_count']
                processing_stats['total_duplicates'] += template_stats['duplicate_count']
                
                # 根据情况显示不同的消息
                if new_pairs_count == 0 and template_stats['generated_count'] > 0:
                    print(f"[{i}/{len(templates)}] ⚠️  生成 {template_stats['generated_count']} 条查询但全部重复")
                elif template_stats['generated_count'] > 5000:
                    print(f"[{i}/{len(templates)}] ⚡ 大量生成 {new_pairs_count} 条新查询 (总生成{template_stats['generated_count']}条)")
                else:
                    print(f"[{i}/{len(templates)}] ✅ 成功生成 {new_pairs_count} 条新查询")
            else:
                processing_stats['failed_templates'] += 1
                processing_stats['failed_details'].append({
                    'template_index': i,
                    'template': template[:100] + '...' if len(template) > 100 else template,
                    'error_stage': template_stats['error_stage'],
                    'error_message': template_stats['error_message']
                })
                print(f"[{i}/{len(templates)}] ❌ 处理失败: {template_stats['error_message']}")
            
        except Exception as e:
            # 处理单个模板时的意外错误
            processing_stats['failed_templates'] += 1
            processing_stats['failed_details'].append({
                'template_index': i,
                'template': template[:100] + '...' if len(template) > 100 else template,
                'error_stage': 'unexpected_error',
                'error_message': str(e)
            })
            print(f"[{i}/{len(templates)}] ❌ 意外错误: {str(e)}")
            continue  # 继续处理下一个模板

    # 转换为列表格式以便后续处理
    full_corpus = [(query, desc) for query, desc in global_corpus_dict.items()]
    
    # --- Save the full corpus to a local file ---
    # 确保只保存成功生成的查询-描述对
    from datetime import datetime
    
    corpus_data = {
        "metadata": {
            "total_templates": len(templates),
            "successful_templates": processing_stats['successful_templates'],
            "failed_templates": processing_stats['failed_templates'],
            "total_unique_queries": len(full_corpus),
            "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "corpus": [
            {
                "query": query,
                "description": desc
            }
            for query, desc in full_corpus
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(corpus_data, f, ensure_ascii=False, indent=2)

    # --- Generate statistics and display results ---
    stats = _generate_statistics(templates, full_corpus, output_file)
    stats.update({
        "total_templates": len(templates),
        "successful_templates": processing_stats['successful_templates'],
        "failed_templates": processing_stats['failed_templates'],
        "output_file": output_file
    })
    _display_final_results(full_corpus, stats)
    
    return {
        "total_templates": len(templates),
        "successful_templates": processing_stats['successful_templates'],
        "failed_templates": processing_stats['failed_templates'],
        "total_unique_queries": len(full_corpus),
        "output_file": output_file,
        "statistics": stats
    }

def _generate_statistics(templates: list, full_corpus: list, output_file: str) -> dict:
    """生成统计信息"""
    # 按查询长度分类统计
    length_stats = {}
    for query, _ in full_corpus:
        steps = query.count('.') 
        length_stats[steps] = length_stats.get(steps, 0) + 1
    
    # 按操作类型分类统计
    operation_stats = {
        "查询(V/E)": 0,
        "创建(addV/addE)": 0, 
        "更新(property)": 0,
        "删除(drop)": 0
    }
    
    for query, _ in full_corpus:
        if query.startswith('g.V(') or query.startswith('g.E('):
            if '.drop()' in query:
                operation_stats["删除(drop)"] += 1
            elif '.property(' in query:
                operation_stats["更新(property)"] += 1
            else:
                operation_stats["查询(V/E)"] += 1
        elif '.addV(' in query or '.addE(' in query:
            operation_stats["创建(addV/addE)"] += 1
    
    return {
        "length_stats": length_stats,
        "operation_stats": operation_stats,
        "avg_per_template": len(full_corpus) / len(templates) if templates else 0
    }

def _display_final_results(full_corpus: list, stats: dict):
    """显示最终生成结果和统计信息"""
    print(f"\n{'='*50}")
    print(f"📊 生成完成统计")
    print(f"{'='*50}")
    print(f"处理的模板数量: {stats.get('total_templates', 0)}")
    print(f"成功处理: {stats.get('successful_templates', 0)}")
    print(f"处理失败: {stats.get('failed_templates', 0)}")
    print(f"生成的独特查询数量: {len(full_corpus)}")
    print(f"语料库已保存到: {stats.get('output_file', 'generated_corpus.json')}")
        
    # 按查询长度分类统计
    print(f"\n{'='*50}")
    print("📈 查询复杂度分析:")
    print(f"{'='*50}")
    
    for steps in sorted(stats['length_stats'].keys()):
        print(f"  {steps}步查询: {stats['length_stats'][steps]} 个")
        
    # 按操作类型分类统计
    print(f"\n{'='*50}")
    print("🔍 操作类型分析:")
    print(f"{'='*50}")
    
    for op_type, count in stats['operation_stats'].items():
        percentage = (count / len(full_corpus)) * 100 if full_corpus else 0
        print(f"  {op_type}: {count} 个 ({percentage:.1f}%)")
        
    print(f"\n{'='*50}")
    print(f"✅ 生成完成！共生成 {len(full_corpus)} 个独特查询")
    print(f"{'='*50}")



if __name__ == '__main__':
    # templates = [
    #     # === 查询操作 (Query) - 40% ===
        
    #     # 基础查询
    #     "g.V().has('name', 'John')",
    #     "g.V().has('title', 'The Matrix')",
    #     "g.V().has('born', 1961)",
    #     "g.V().hasLabel('person')",
    #     "g.V().hasLabel('movie')",
        
    #     # 导航查询
    #     "g.V().has('name', 'Laurence Fishburne').out('acted_in')",
    #     "g.V().has('title', 'The Matrix').in('acted_in')",
    #     "g.V().hasLabel('person').out('directed')",
    #     "g.V().hasLabel('movie').in('rate')",
        
    #     # 复杂查询
    #     "g.V().has('name', 'Laurence Fishburne').out('acted_in').has('title', 'The Matrix')",
    #     "g.V().hasLabel('person').out('acted_in').in('rate')",
    #     "g.V().has('title', 'Matrix').in('acted_in').out('directed')",
        
    #     # === 创建操作 (Create) - 25% ===
        
    #     # 基础创建
    #     "g.addV('person')",
    #     "g.addV('movie')",
    #     "g.addV('user')",
        
    #     # 带属性创建
    #     "g.addV('person').property('name', 'New Actor')",
    #     "g.addV('movie').property('title', 'New Movie')",
    #     "g.addV('person').property('name', 'Jane').property('born', 1990)",
    #     "g.addV('movie').property('title', 'Test Movie').property('duration', 120)",
    #     "g.addV('user').property('login', 'newuser').property('name', 'New User')",
        
    #     # === 更新操作 (Update) - 25% ===
        
    #     # 单属性更新
    #     "g.V().has('name', 'John').property('born', 1990)",
    #     "g.V().has('title', 'Test').property('duration', 120)",
    #     "g.V().hasLabel('person').has('name', 'Jane').property('born', 1985)",
    #     "g.V().hasLabel('movie').has('title', 'Old Movie').property('rated', 'PG-13')",
        
    #     # 多属性更新
    #     "g.V().has('name', 'John').property('born', 1990).property('poster_image', 'new_url')",
    #     "g.V().has('title', 'Test').property('duration', 150).property('rated', 'R')",
    #     "g.V().hasLabel('user').has('login', 'testuser').property('name', 'Updated Name').property('born', 1995)",
        
    #     # === 删除操作 (Delete) - 10% ===
        
    #     # 基础删除
    #     "g.V().has('name', 'temp_person').drop()",
    #     "g.V().has('title', 'temp_movie').drop()",
    #     "g.V().hasLabel('user').has('login', 'temp_user').drop()",
        
    #     # 条件删除
    #     "g.V().hasLabel('person').has('born', 0).drop()",
    #     "g.V().hasLabel('movie').has('duration', 0).drop()",
    # ]
    
    def load_templates_from_csv(csv_file_path: str) -> tuple[list[str], dict]:
        """
        从CSV文件中加载Gremlin查询作为模板
        
        Args:
            csv_file_path: CSV文件路径
            
        Returns:
            tuple: (成功加载的查询列表, 统计信息字典)
        """
        import csv
        
        templates = []
        stats = {
            'total_rows': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'failed_queries': []
        }
        
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row_num, row in enumerate(reader, 1):
                    stats['total_rows'] += 1
                    
                    try:
                        # 获取gremlin_query列
                        gremlin_query = row.get('gremlin_query', '').strip()
                        
                        if not gremlin_query:
                            stats['failed_loads'] += 1
                            stats['failed_queries'].append(f"第{row_num}行: 空查询")
                            continue
                        
                        # 移除可能的引号包围
                        if gremlin_query.startswith('"') and gremlin_query.endswith('"'):
                            gremlin_query = gremlin_query[1:-1]
                        
                        # 基本语法检查
                        if not gremlin_query.startswith('g.'):
                            stats['failed_loads'] += 1
                            stats['failed_queries'].append(f"第{row_num}行: 格式错误")
                            continue
                        
                        templates.append(gremlin_query)
                        stats['successful_loads'] += 1
                            
                    except Exception as e:
                        stats['failed_loads'] += 1
                        stats['failed_queries'].append(f"第{row_num}行: {str(e)}")
                        continue
                        
        except FileNotFoundError:
            print(f"❌ 错误: 找不到CSV文件: {csv_file_path}")
            return [], stats
        except Exception as e:
            print(f"❌ 读取CSV文件时发生错误: {str(e)}")
            return [], stats
        
        return templates, stats
    
    # 从CSV文件加载模板
    csv_file_path = "cypher2gremlin_dataset.csv"
    
    print(f"🔄 从 {csv_file_path} 加载Gremlin查询模板...")
    templates, load_stats = load_templates_from_csv(csv_file_path)
    
    print(f"📊 CSV加载统计: {load_stats['successful_loads']}/{load_stats['total_rows']} 成功")
    
    if load_stats['failed_loads'] > 0:
        print(f"⚠️  {load_stats['failed_loads']} 个模板加载失败")
    
    if not templates:
        print("❌ 没有成功加载任何模板，程序退出")
        exit(1)
    
    print(f"✅ 成功加载 {len(templates)} 个模板，开始生成语料库...")
    
    # 生成语料库
    try:
        result = generate_corpus_from_templates(templates)
    except Exception as e:
        print(f"❌ 生成过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()