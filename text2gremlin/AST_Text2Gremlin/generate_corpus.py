# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


"""
Gremlin 查询语料库生成脚本

从模板生成 Gremlin 查询语料库的命令行工具。

用法:
    # 使用默认配置（推荐）
    python generate_corpus.py
    
    # 自定义参数
    python generate_corpus.py --templates my_templates.csv --num-queries 50
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# 添加 base 包到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from base import generate_gremlin_corpus


def load_config(config_path='config.json'):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  警告: 无法加载配置文件 {config_path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description='生成 Gremlin 查询语料库',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python generate_corpus.py
  
  # 使用自定义模板文件
  python generate_corpus.py --templates my_templates.csv
  
  # 使用自定义配置文件
  python generate_corpus.py --config my_config.json
  
  # 完全自定义
  python generate_corpus.py --templates templates.csv --schema schema.json --data data/ --output output.json

配置说明:
  config.json 中的配置项：
  - templates_file: 模板文件路径（默认: gremlin_templates.csv）
  - db_schema_path: schema 文件路径
  - data_path: 数据目录路径（默认: db_data/）
  - output_dir: 输出目录（默认: output）
  
  查询数量控制:
  - 由 combination_control_config.json 中的 max_total_combinations 控制
  - 根据查询复杂度自动调整（short/medium/long/ultra）

注意:
  - 输出文件自动命名为 output/generated_corpus_YYYYMMDD_HHMMSS.json
  - 每次运行生成新文件，不会覆盖旧文件
        """
    )
    
    parser.add_argument(
        '--config',
        default='config.json',
        help='配置文件路径 (JSON格式，默认: config.json)'
    )
    
    parser.add_argument(
        '--templates', 
        help='模板文件路径 (CSV格式，默认从 config.json 读取)'
    )
    
    parser.add_argument(
        '--schema',
        help='图数据库模式文件路径 (JSON格式，默认从 config.json 读取)'
    )
    
    parser.add_argument(
        '--data',
        help='数据目录路径 (默认从 config.json 读取)'
    )
    
    parser.add_argument(
        '--output',
        help='输出文件路径 (JSON格式，默认: output/generated_corpus_YYYYMMDD_HHMMSS.json)'
    )
    

    
    args = parser.parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 从配置文件或命令行参数获取值（命令行参数优先）
    templates_file = args.templates or config.get('templates_file', 'gremlin_templates.csv')
    db_id = config.get('db_id', 'movie')
    schema_path = args.schema or config.get('db_schema_path', {}).get(db_id, 'db_data/schema/movie_schema.json')
    data_path = args.data or config.get('data_path', 'db_data/')
    output_dir = config.get('output_dir', 'output')
    
    # 更新 args 对象
    args.templates = templates_file
    args.schema = schema_path
    args.data = data_path
    
    # 验证输入文件
    if not os.path.exists(args.templates):
        print(f"❌ 错误: 模板文件不存在: {args.templates}")
        print(f"💡 提示: 请创建 {args.templates} 文件，或使用 --templates 指定其他文件")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"❌ 错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.schema):
        print(f"❌ 错误: 模式文件不存在: {args.schema}")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"❌ 错误: 数据目录不存在: {args.data}")
        sys.exit(1)
    
    # 如果没有指定输出文件，使用默认路径
    if not args.output:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'{output_dir}/generated_corpus_{timestamp}.json'
    
    try:
        print("=" * 60)
        print("🚀 Gremlin 查询语料库生成器")
        print("=" * 60)
        print(f"\n📋 配置信息:")
        print(f"  模板文件: {args.templates}")
        print(f"  配置文件: {args.config}")
        print(f"  模式文件: {args.schema}")
        print(f"  数据目录: {args.data}")
        print(f"  输出文件: {args.output}")
        
        print("\n" + "-" * 60)
        
        # 调用生成器
        result = generate_gremlin_corpus(
            templates=args.templates,
            config_path=args.config,
            schema_path=args.schema,
            data_path=args.data,
            output_file=args.output
        )
        
        print("\n" + "=" * 60)
        print("✅ 生成完成！")
        print("=" * 60)
        print(f"\n📊 统计信息:")
        print(f"  总模板数: {result['total_templates']}")
        print(f"  成功处理: {result['successful_templates']}")
        print(f"  处理失败: {result['failed_templates']}")
        print(f"  生成查询数: {result['total_unique_queries']}")
        
        if 'output_file' in result:
            print(f"\n💾 结果已保存到: {result['output_file']}")
            print(f"\n💡 提示:")
            print(f"  - 可以在 {args.templates} 中添加更多模板")
            print(f"  - 查询数量由 combination_control_config.json 控制")
        else:
            print(f"\n生成了 {len(result['queries'])} 个查询 (未保存到文件)")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
