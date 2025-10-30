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


import os
import random
import string
import json
from typing import List, Dict, Any, Tuple, Set, Optional

from .Schema import Schema
from .GremlinParse import Traversal, Step
from .GremlinExpr import Predicate
from .GremlinBase import GremlinBase
from .Config import Config
from .CombinationController import CombinationController


class TraversalGenerator:
    """Gremlin查询生成器 - 分层泛化架构"""
    # A. 简单步骤（无参数，直接生成）
    SIMPLE_STEPS = {
        'count': {'output_type': 'number'},
        'id': {'output_type': 'value'},
        'label': {'output_type': 'string'},
        'fold': {'output_type': 'list'},
        'unfold': {'output_type': None},  
        'drop': {'output_type': 'none'},
        'iterate': {'output_type': 'none'},
        'explain': {'output_type': 'string'}, 
        'profile': {'output_type': 'map'}, 
        'loops': {'output_type': 'number'},  
        'value': {'output_type': 'value'}, 
        'identity': {'output_type': None},  
        'barrier': {'output_type': None}  
    }
    
    # B. 属性访问步骤（需要Schema + 泛化）
    PROPERTY_ACCESS_STEPS = {
        'values': {'output_type': 'value', 'supports_params': True},
        'properties': {'output_type': 'property', 'supports_params': True},
        'valueMap': {'output_type': 'map', 'supports_params': True},
        'elementMap': {'output_type': 'map', 'supports_params': True},
        'key': {'output_type': 'string', 'supports_params': False}
    }
    
    # C. 数值参数步骤
    NUMERIC_PARAM_STEPS = {
        'limit': {'range': (1, 100)},
        'skip': {'range': (0, 50)},
        'tail': {'range': (1, 20)},
        'sample': {'range': (1, 10)},
        'range': {'range': (0, 100)},
        'coin': {'range': (0.1, 0.9), 'type': 'float'}  # 概率值，0.0-1.0
    }
    
    # D. 导航步骤（需要Schema关系 + 泛化）
    NAVIGATION_STEPS = {
        'out': {'direction': 'outgoing', 'target': 'vertex'},
        'in': {'direction': 'incoming', 'target': 'vertex'},
        'both': {'direction': 'both', 'target': 'vertex'},
        'outE': {'direction': 'outgoing', 'target': 'edge'},
        'inE': {'direction': 'incoming', 'target': 'edge'},
        'bothE': {'direction': 'both', 'target': 'edge'},
        'outV': {'direction': 'out', 'target': 'vertex'},
        'inV': {'direction': 'in', 'target': 'vertex'},
        'bothV': {'direction': 'both', 'target': 'vertex'},  # 从边到两端顶点
        'otherV': {'direction': 'other', 'target': 'vertex'}
    }
    

    # E. 过滤步骤（需要条件 + 泛化）
    FILTER_STEPS = {
        'has': {'type': 'property_filter', 'needs_value': True},
        'hasLabel': {'type': 'label_filter', 'needs_value': True},
        'hasId': {'type': 'id_filter', 'needs_value': True},
        'hasKey': {'type': 'key_filter', 'needs_value': True},
        'hasValue': {'type': 'value_filter', 'needs_value': True},
        'where': {'type': 'complex_filter', 'needs_traversal': True},
        'is': {'type': 'value_comparison', 'needs_value': True},
        'not': {'type': 'negation', 'needs_traversal': True},
        'as': {'type': 'label_marker', 'needs_value': True},
        'filter': {'type': 'traversal_filter', 'needs_traversal': True},  # 通用过滤
        'and': {'type': 'logical_and', 'needs_traversal': True, 'multi_param': True},  # 逻辑与
        'or': {'type': 'logical_or', 'needs_traversal': True, 'multi_param': True}  # 逻辑或
    }
    
    # F. 转换步骤
    TRANSFORM_STEPS = {
        'order': {'output_type': None},
        'dedup': {'output_type': None},
        'simplePath': {'output_type': None},
        'cyclicPath': {'output_type': None},
        'by': {'output_type': None, 'is_modulator': True},  # 修饰符步骤
        'constant': {'output_type': 'value', 'needs_value': True}  # 常量映射
    }
    
    # G. 聚合步骤
    AGGREGATE_STEPS = {
        'group': {'output_type': 'map'},
        'groupCount': {'output_type': 'map'},
        'sum': {'output_type': 'number'},
        'mean': {'output_type': 'number'},
        'min': {'output_type': 'value'},
        'max': {'output_type': 'value'}
    }
    
    # G2. 副作用步骤（Side Effect）
    SIDE_EFFECT_STEPS = {
        'aggregate': {'output_type': None, 'needs_key': True},  # 聚合到侧效果
        'store': {'output_type': None, 'needs_key': True},  # 存储到侧效果
        'sideEffect': {'output_type': None, 'needs_traversal': True},  # 执行侧效果遍历
        'cap': {'output_type': 'value', 'needs_key': True},  # 获取侧效果值
        'sack': {'output_type': 'value'}  # 获取sack值
    }
    
    # H. 终端步骤
    TERMINAL_STEPS = {
        'toList': {'output_type': 'list'},
        'toSet': {'output_type': 'set'},
        'next': {'output_type': 'value'},
        'hasNext': {'output_type': 'boolean'},
        'tryNext': {'output_type': 'optional'}  # 返回 Optional
    }
    
    
    # I. 边修改步骤
    EDGE_MODIFICATION_STEPS = {
        'from': {'output_type': None, 'needs_label_or_traversal': True},
        'to': {'output_type': None, 'needs_label_or_traversal': True}
    }
    
    # J. 谓词（用于has等步骤）
    # 谓词由 Visitor 解析为 Predicate/TextPredicate 对象，在 E 层过滤步骤中处理
    PREDICATES = {
        # 数值谓词
        'eq': {'types': ['numeric', 'string', 'any']},
        'neq': {'types': ['numeric', 'string', 'any']},
        'gt': {'types': ['numeric']},
        'gte': {'types': ['numeric']},
        'lt': {'types': ['numeric']},
        'lte': {'types': ['numeric']},
        'between': {'types': ['numeric']},
        'inside': {'types': ['numeric']},
        'outside': {'types': ['numeric']},
        # 字符串谓词（TextP）
        'startingWith': {'types': ['string']},
        'endingWith': {'types': ['string']},
        'containing': {'types': ['string']},
        'notStartingWith': {'types': ['string']},
        'notEndingWith': {'types': ['string']},
        'notContaining': {'types': ['string']},
        'regex': {'types': ['string']},
        'notRegex': {'types': ['string']},
        # 集合谓词
        'within': {'types': ['any']},
        'without': {'types': ['any']},
        # 否定谓词
        'not': {'types': ['any']}
    }
    

    # K. 图算法步骤
    GRAPH_ALGORITHM_STEPS = {
        'pageRank': {'output_type': None},
        'peerPressure': {'output_type': None},
        'connectedComponent': {'output_type': None},
        'shortestPath': {'output_type': None}
    }
    
    # L. 工具步骤
    UTILITY_STEPS = {
        'math': {'output_type': 'number', 'needs_expression': True},
        'subgraph': {'output_type': None, 'needs_key': True},
        'timeLimit': {'output_type': None, 'needs_number': True},
        'inject': {'output_type': None, 'multi_param': True},  # 支持多参数
        'call': {'output_type': None, 'needs_string': True},
        'io': {'output_type': None, 'needs_string': True},
        'mergeE': {'output_type': None},  # 合并边
        'mergeV': {'output_type': None},  # 合并顶点
        'with': {'output_type': None, 'multi_param': True}  # 配置选项
    }
    
    # M. 特殊步骤（需要单独实现）
    SPECIAL_STEPS = {
        # 起始步骤
        'V': {'category': 'start'},
        'E': {'category': 'start'},
        # 写操作
        'addV': {'category': 'write'},
        'addE': {'category': 'write'},
        'property': {'category': 'write'},
        # 分支逻辑
        'choose': {'category': 'branch'},
        'coalesce': {'category': 'branch'},
        'optional': {'category': 'branch'},
        # 循环逻辑
        'repeat': {'category': 'loop'},
        'until': {'category': 'loop'},
        'times': {'category': 'loop'},
        'emit': {'category': 'loop'},
        # 模式匹配
        'match': {'category': 'pattern'},
        # 投影
        'select': {'category': 'projection'},
        'project': {'category': 'projection'},
        # 路径
        'path': {'category': 'path'},
        'tree': {'category': 'path'},
        # 高阶操作
        'union': {'category': 'higher_order'},
        'flatMap': {'category': 'higher_order'},
        'map': {'category': 'higher_order'},
        'local': {'category': 'higher_order'}  # 本地作用域遍历
    }
    
    def __init__(self, schema: Schema, recipe: Traversal, gremlin_base: GremlinBase, 
                 controller: Optional[CombinationController] = None):
        """
        初始化生成器
        
        Args:
            schema: 图Schema
            recipe: 查询配方
            gremlin_base: Gremlin基础工具
            controller: 组合控制器（可选）
        """
        self.schema = schema
        self.recipe = recipe
        self.gremlin_base = gremlin_base
        self.generated_pairs: Set[Tuple[str, str]] = set()
        
        # 集成组合控制器
        if controller is None:
            try:
                possible_paths = [
                    'combination_control_config.json',  
                    os.path.join(os.path.dirname(__file__), 'combination_control_config.json'),  
                ]
                
                config_loaded = False
                for config_path in possible_paths:
                    if os.path.exists(config_path):
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        self.controller = CombinationController(config)
                        config_loaded = True
                        break
                
                if not config_loaded:
                    print(f"⚠️  未找到CombinationController配置文件")
                    self.controller = None
                    
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"⚠️  无法加载CombinationController配置: {e}")
                self.controller = None
        else:
            self.controller = controller
        
        # 随机增强计数器（跟踪每个查询路径的增强次数）
        self.enhancement_counts: Dict[str, int] = {}
        
        # 配方路径完成标记
        self.recipe_path_completed = False
    
    
    def generate(self) -> List[Tuple[str, str]]:
        """
        主生成函数，启动递归生成过程
        
        保护机制：
        1. 优先完成配方路径（忽略数量限制）
        2. 配方完成后，才应用数量限制
        3. 如果最终超出限制，优先保留深层查询
        
        Returns:
            去重后的 (查询, 描述) 对列表
        """
        self.generated_pairs.clear()
        self.enhancement_counts.clear()
        self.recipe_path_completed = False
        
        self._recursive_generate(
            recipe_steps=self.recipe.steps,
            current_query="g",
            current_desc="从图中开始",
            current_label=None,
            current_type='graph'
        )
        
        # 后处理：如果超出限制，优先保留深层查询
        results = list(self.generated_pairs)
        
        if self.controller:
            category = self.controller.get_chain_category(len(self.recipe.steps))
            max_limit = self.controller.max_total.get(category)
            
            if max_limit and len(results) > max_limit:
                # 超出限制，按深度排序，保留深层查询
                def get_depth(query_desc_pair):
                    query = query_desc_pair[0]
                    # 统计步骤数（排除字符串中的点号）
                    # 简单方法：统计方法调用（大写字母开头或小写方法名）
                    import re
                    steps = re.findall(r'\.\w+\(', query)
                    return len(steps)
                
                # 按深度降序排序
                results.sort(key=get_depth, reverse=True)
                
                # 保留前max_limit个
                results = results[:max_limit]
                
                print(f"⚠️  生成数量超出限制，已裁剪：{len(self.generated_pairs)} → {max_limit}")
        
        return results
    
    def _recursive_generate(self, recipe_steps: List[Step], current_query: str, 
                           current_desc: str, current_label: str, current_type: str):
        """
        核心递归生成函数 - 深度优先搜索 + 回溯
        
        Args:
            recipe_steps: 剩余的配方步骤
            current_query: 当前查询字符串
            current_desc: 当前描述字符串
            current_label: 当前顶点/边标签
            current_type: 当前类型 (graph/vertex/edge/value/...)
        """
        # 1. 终止条件：配方步骤用完
        if not recipe_steps:
            self.generated_pairs.add((current_query, current_desc))
            
            # 标记配方路径完成（第一次到达终点）
            if not self.recipe_path_completed:
                self.recipe_path_completed = True
                print(f"✅ 配方路径完成: {current_query}")
            
            # 尝试随机增强（终端步骤）
            if self.controller:
                enhancement_count = self.enhancement_counts.get(current_query, 0)
                if self.controller.should_apply_random_enhancement(True, enhancement_count):
                    enhanced = self._apply_random_enhancement(
                        current_query, current_desc, current_label, current_type
                    )
                    for enh_query, enh_desc in enhanced:
                        self.generated_pairs.add((enh_query, enh_desc))
                        self.enhancement_counts[enh_query] = enhancement_count + 1
            return
        
        # 2. 取出当前步骤
        step_recipe = recipe_steps[0]
        remaining_steps = recipe_steps[1:]
        
        # 3. 检查是否应该停止生成（保护机制）
        if self.controller:
            category = self.controller.get_chain_category(len(self.recipe.steps))
            
            # 如果配方路径未完成，不停止（保护机制）
            if self.recipe_path_completed:
                if self.controller.should_stop_generation(len(self.generated_pairs), category):
                    return
        
        # 4. 获取当前步骤的所有合法选项
        options = self._get_valid_options_for_step(
            step_recipe, current_label, current_type, remaining_steps
        )
        
        # 5. 遍历每个选项，继续递归
        for option in options:
            next_query = current_query + option['query_part']
            next_desc = current_desc + option['desc_part']
            
            # 保存中间结果
            self.generated_pairs.add((next_query, next_desc))
            
            # 尝试随机增强（中间步骤）
            if self.controller and remaining_steps:
                enhancement_count = self.enhancement_counts.get(next_query, 0)
                if self.controller.should_apply_random_enhancement(False, enhancement_count):
                    enhanced = self._apply_random_enhancement(
                        next_query, next_desc, option['new_label'], option['new_type']
                    )
                    for enh_query, enh_desc in enhanced:
                        # 对增强后的查询继续执行剩余步骤
                        self._recursive_generate(
                            remaining_steps, enh_query, enh_desc, 
                            option['new_label'], option['new_type']
                        )
                        self.enhancement_counts[enh_query] = enhancement_count + 1
            
            # 继续递归（如果不是终止类型）
            if option['new_type'] != 'none':
                self._recursive_generate(
                    remaining_steps, next_query, next_desc,
                    option['new_label'], option['new_type']
                )
    
    #步骤选项生成（分发器）
    
    def _get_valid_options_for_step(self, step_recipe: Step, current_label: str, 
                                    current_type: str, remaining_steps: List[Step] = None) -> List[Dict]:
        """
        根据步骤类型分发到对应的处理器
        
        Args:
            step_recipe: 配方步骤
            current_label: 当前标签
            current_type: 当前类型
            remaining_steps: 剩余步骤（用于判断是否是终端步骤）
            
        Returns:
            选项列表，每个选项包含: query_part, desc_part, new_label, new_type
        """
        step_name = step_recipe.name
        
        # 优先级1: 特殊步骤（复杂逻辑）
        if step_name in self.SPECIAL_STEPS:
            return self._handle_special_step(step_recipe, current_label, current_type, remaining_steps)
        
        # 优先级2: 过滤步骤（需要泛化）
        if step_name in self.FILTER_STEPS:
            return self._handle_filter_step(step_recipe, current_label, current_type, remaining_steps)
        
        # 优先级3: 导航步骤（需要Schema）
        if step_name in self.NAVIGATION_STEPS:
            return self._handle_navigation_step(step_recipe, current_label, current_type)
        
        # 优先级4: 属性访问步骤
        if step_name in self.PROPERTY_ACCESS_STEPS:
            return self._handle_property_access_step(step_recipe, current_label, current_type)
        
        # 优先级5: 简单步骤
        if step_name in self.SIMPLE_STEPS:
            return self._handle_simple_step(step_name, current_label, current_type)
        
        # 优先级6: 转换步骤
        if step_name in self.TRANSFORM_STEPS:
            return self._handle_transform_step(step_recipe, current_label, current_type)
        
        # 优先级7: 聚合步骤
        if step_name in self.AGGREGATE_STEPS:
            return self._handle_aggregate_step(step_recipe, current_label, current_type)
        
        # 优先级7.5: 副作用步骤
        if step_name in self.SIDE_EFFECT_STEPS:
            return self._handle_side_effect_step(step_recipe, current_label, current_type)
        
        # 优先级8: 终端步骤
        if step_name in self.TERMINAL_STEPS:
            return self._handle_terminal_step(step_recipe, current_label, current_type)
        
        # 优先级9: 数值参数步骤
        if step_name in self.NUMERIC_PARAM_STEPS:
            return self._handle_numeric_param_step(step_name, step_recipe.params, current_label, current_type)
        
        # 优先级10: 图算法步骤
        if step_name in self.GRAPH_ALGORITHM_STEPS:
            return self._handle_graph_algorithm_step(step_name, current_label, current_type)
        
        # 优先级11: 工具步骤
        if step_name in self.UTILITY_STEPS:
            return self._handle_utility_step(step_recipe, current_label, current_type)
        
        # 优先级12: 边修改步骤
        if step_name in self.EDGE_MODIFICATION_STEPS:
            return self._handle_edge_modification_step(step_recipe, current_label, current_type)
        
        # 未知步骤
        print(f"⚠️  未知步骤: {step_name}")
        return []
    
    #   A. 简单步骤处理器  
    
    def _handle_simple_step(self, step_name: str, current_label: str, 
                           current_type: str) -> List[Dict]:
        """
        处理简单步骤（无参数）
        
        Args:
            step_name: 步骤名称
            current_label: 当前标签
            current_type: 当前类型
            
        Returns:
            选项列表
        """
        step_info = self.SIMPLE_STEPS[step_name]
        new_type = step_info['output_type']
        
        # 如果new_type是None，保持当前类型
        if new_type is None:
            new_type = current_type
        
        # 从GremlinBase获取翻译
        desc = self.gremlin_base.get_token_desc(step_name)
        
        return [{
            'query_part': f'.{step_name}()',
            'desc_part': f'，{desc}',
            'new_label': current_label,
            'new_type': new_type
        }]
    
    #   F. 转换步骤处理器  
    
    def _handle_transform_step(self, step_recipe: Step, current_label: str,
                               current_type: str) -> List[Dict]:
        """
        处理转换步骤（order, dedup等）
        
        转换步骤通常不改变数据的结构，只是对数据进行排序、去重、限制等操作
        
        Args:
            step_recipe: 配方步骤
            current_label: 当前标签
            current_type: 当前类型
            
        Returns:
            选项列表
        """
        step_name = step_recipe.name
        step_params = step_recipe.params
        step_info = self.TRANSFORM_STEPS[step_name]
        new_type = step_info['output_type']
        
        if new_type is None:
            new_type = current_type
        
        options = []
        
        # 从GremlinBase获取翻译
        desc = self.gremlin_base.get_token_desc(step_name)
        
        # order() 步骤 - 排序
        if step_name == 'order':
            # order() 不接受参数，必须用 .by() 修饰符
            # 如果配方中有参数，说明是错误的语法，忽略参数
            options.append({
                'query_part': f'.{step_name}()',
                'desc_part': f'，{desc}',
                'new_label': current_label,
                'new_type': new_type
            })
        
        # dedup() 步骤 - 去重
        elif step_name == 'dedup':
            # dedup() 可以接受标签参数（用于 as/select 场景）
            # dedup('a') - 按标签 'a' 去重
            # dedup() - 去重整个元素
            # 注意：如果要按属性去重，应该用 .dedup().by('name')
            if step_params:
                # 有参数：这是标签参数，保留原样
                params_str = ', '.join([f"'{p}'" if isinstance(p, str) else str(p) for p in step_params])
                options.append({
                    'query_part': f'.{step_name}({params_str})',
                    'desc_part': f'，{desc}（按标签）',
                    'new_label': current_label,
                    'new_type': new_type
                })
            else:
                # 无参数
                options.append({
                    'query_part': f'.{step_name}()',
                    'desc_part': f'，{desc}',
                    'new_label': current_label,
                    'new_type': new_type
                })
        
        # simplePath() 和 cyclicPath() - 路径过滤
        elif step_name in ['simplePath', 'cyclicPath']:
            # 这些步骤通常无参数
            options.append({
                'query_part': f'.{step_name}()',
                'desc_part': f'，{desc}',
                'new_label': current_label,
                'new_type': new_type
            })
        
        # by() 步骤 - 修饰符
        elif step_name == 'by':
            # by() 是修饰符步骤，用于修饰 order(), dedup(), group() 等
            # 通常有参数，保留原样
            if step_params:
                # 格式化参数
                params_str = ', '.join([f"'{p}'" if isinstance(p, str) else str(p) if p is not None else '' for p in step_params])
                # 移除尾部的逗号和空格
                params_str = params_str.rstrip(', ')
                options.append({
                    'query_part': f'.{step_name}({params_str})',
                    'desc_part': f'，{desc}',
                    'new_label': current_label,
                    'new_type': new_type
                })
            else:
                # 无参数
                options.append({
                    'query_part': f'.{step_name}()',
                    'desc_part': f'，{desc}',
                    'new_label': current_label,
                    'new_type': new_type
                })
        
        # constant() 步骤 - 常量映射
        elif step_name == 'constant':
            # constant接受一个值参数
            if step_params:
                value = step_params[0]
                value_str = f"'{value}'" if isinstance(value, str) else str(value)
                options.append({
                    'query_part': f'.{step_name}({value_str})',
                    'desc_part': f'，{desc}（值：{value}）',
                    'new_label': current_label,
                    'new_type': new_type
                })
            else:
                # 无参数，生成一个默认值
                options.append({
                    'query_part': f'.{step_name}("value")',
                    'desc_part': f'，{desc}',
                    'new_label': current_label,
                    'new_type': new_type
                })
        
        # 其他转换步骤
        else:
            options.append({
                'query_part': f'.{step_name}()',
                'desc_part': f'，{desc}',
                'new_label': current_label,
                'new_type': new_type
            })
        
        return options
    
    #   G. 聚合步骤处理器  
    
    def _handle_aggregate_step(self, step_recipe: Step, current_label: str,
                               current_type: str) -> List[Dict]:
        """
        处理聚合步骤（group, sum等）
        
        聚合步骤通常将数据转换为聚合结果（如 map、number 等）
        
        Args:
            step_recipe: 配方步骤
            current_label: 当前标签
            current_type: 当前类型
            
        Returns:
            选项列表
        """
        step_name = step_recipe.name
        step_params = step_recipe.params
        step_info = self.AGGREGATE_STEPS[step_name]
        new_type = step_info['output_type']
        
        # 从GremlinBase获取翻译
        desc = self.gremlin_base.get_token_desc(step_name)
        
        # group() 和 groupCount() 可以接受标签参数（用于 side-effect）
        if step_name in ['group', 'groupCount'] and step_params:
            # 有参数：这是标签参数，保留原样
            params_str = ', '.join([f"'{p}'" if isinstance(p, str) else str(p) for p in step_params])
            return [{
                'query_part': f'.{step_name}({params_str})',
                'desc_part': f'，{desc}（副作用）',
                'new_label': None,
                'new_type': new_type
            }]
        else:
            # 无参数版本
            return [{
                'query_part': f'.{step_name}()',
                'desc_part': f'，{desc}',
                'new_label': None,  # 聚合后通常没有标签
                'new_type': new_type
            }]
    
    #   G2. 副作用步骤处理器  
    
    def _handle_side_effect_step(self, step_recipe: Step, current_label: str,
                                 current_type: str) -> List[Dict]:
        """
        处理副作用步骤（aggregate, store, sideEffect等）
        
        副作用步骤通常不改变遍历流，但会产生副作用
        
        Args:
            step_recipe: 配方步骤
            current_label: 当前标签
            current_type: 当前类型
            
        Returns:
            选项列表
        """
        from .GremlinExpr import AnonymousTraversal
        
        step_name = step_recipe.name
        step_params = step_recipe.params
        step_info = self.SIDE_EFFECT_STEPS[step_name]
        new_type = step_info.get('output_type', current_type)
        
        # aggregate, store, cap需要key参数
        if step_info.get('needs_key'):
            if not step_params:
                # 生成一个默认key
                key = 'x'
            else:
                key = step_params[0]
            
            # 从GremlinBase获取翻译，传入key参数
            desc = self.gremlin_base.get_token_desc(step_name, key)
            
            return [{
                'query_part': f".{step_name}('{key}')",
                'desc_part': f"，{desc}",
                'new_label': current_label,
                'new_type': new_type
            }]
        
        # sideEffect需要traversal参数
        elif step_info.get('needs_traversal'):
            # 从GremlinBase获取翻译
            desc = self.gremlin_base.get_token_desc(step_name)
            
            if not step_params or not isinstance(step_params[0], AnonymousTraversal):
                # 生成一个简单的副作用遍历
                return [{
                    'query_part': f".{step_name}(__.identity())",
                    'desc_part': f"，{desc}",
                    'new_label': current_label,
                    'new_type': new_type
                }]
            else:
                # 使用提供的遍历，递归生成变体
                traversal = step_params[0]
                nested_variants = self._generate_nested_traversal_variants(traversal, current_depth=0)
                
                result = []
                for variant in nested_variants[:1]:  # 只取第一个变体避免组合爆炸
                    result.append({
                        'query_part': f".{step_name}({variant})",
                        'desc_part': f"，{desc}",
                        'new_label': current_label,
                        'new_type': new_type
                    })
                return result if result else [{
                    'query_part': f".{step_name}(__.identity())",
                    'desc_part': f"，{desc}",
                    'new_label': current_label,
                    'new_type': new_type
                }]
        
        # sack无参数
        else:
            # 从GremlinBase获取翻译
            desc = self.gremlin_base.get_token_desc(step_name)
            
            return [{
                'query_part': f".{step_name}()",
                'desc_part': f"，{desc}",
                'new_label': current_label,
                'new_type': new_type
            }]
    
    #   H. 终端步骤处理器  
    
    def _handle_terminal_step(self, step_recipe: Step, current_label: str,
                             current_type: str) -> List[Dict]:
        """
        处理终端步骤（toList, next等）
        
        终端步骤通常是查询的最后一步，用于获取结果
        
        Args:
            step_recipe: 配方步骤
            current_label: 当前标签
            current_type: 当前类型
            
        Returns:
            选项列表
        """
        step_name = step_recipe.name
        step_params = step_recipe.params
        step_info = self.TERMINAL_STEPS[step_name]
        new_type = step_info['output_type']
        
        # 从GremlinBase获取翻译
        desc = self.gremlin_base.get_token_desc(step_name)
        
        # next() 可以接受整数参数：next(n) 获取 n 个元素
        if step_name == 'next' and step_params:
            # 检查参数是否是整数（不是 Terminal 对象）
            from .GremlinExpr import Terminal
            if step_params and not isinstance(step_params[0], Terminal):
                # 有整数参数
                param_val = step_params[0]
                return [{
                    'query_part': f'.{step_name}({param_val})',
                    'desc_part': f'，{desc}（获取{param_val}个）',
                    'new_label': current_label,
                    'new_type': new_type
                }]
        
        # 无参数版本
        return [{
            'query_part': f'.{step_name}()',
            'desc_part': f'，{desc}',
            'new_label': current_label,
            'new_type': new_type
        }]
    
    #   C. 数值参数步骤处理器  
    
    def _handle_numeric_param_step(self, step_name: str, params: List, 
                                   current_label: str, current_type: str) -> List[Dict]:
        """
        处理数值参数步骤（limit, skip等）
        
        Args:
            step_name: 步骤名称
            params: 配方参数
            current_label: 当前标签
            current_type: 当前类型
            
        Returns:
            选项列表
        """
        config = self.NUMERIC_PARAM_STEPS[step_name]
        
        # 如果配方提供了参数，使用配方参数
        if params:
            value = params[0]
        else:
            # 否则在合理范围内随机生成
            min_val, max_val = config['range']
            value = random.randint(min_val, max_val)
        
        # 从GremlinBase获取翻译
        desc = self.gremlin_base.get_token_desc(step_name)
        
        # 如果翻译中包含占位符，替换它
        if '{}' in desc:
            desc = desc.format(value)
        
        return [{
            'query_part': f'.{step_name}({value})',
            'desc_part': f'，{desc}',
            'new_label': current_label,
            'new_type': current_type
        }]
    
    #   K. 图算法步骤处理器  
    
    def _handle_graph_algorithm_step(self, step_name: str, current_label: str,
                                     current_type: str) -> List[Dict]:
        """
        处理图算法步骤（pageRank, connectedComponent等）
        
        这些步骤通常无参数或有配置参数，主要用于图分析
        
        Args:
            step_name: 步骤名称
            current_label: 当前标签
            current_type: 当前类型
            
        Returns:
            选项列表
        """
        # 从GremlinBase获取翻译
        desc = self.gremlin_base.get_token_desc(step_name)
        
        return [{
            'query_part': f".{step_name}()",
            'desc_part': f"，{desc}",
            'new_label': current_label,
            'new_type': current_type
        }]
    
    #   L. 工具步骤处理器  
    
    def _handle_utility_step(self, step_recipe: Step, current_label: str,
                            current_type: str) -> List[Dict]:
        """
        处理工具步骤（math, subgraph, timeLimit, inject, call, io等）
        
        这些步骤有各种不同的参数类型
        
        Args:
            step_recipe: 配方步骤
            current_label: 当前标签
            current_type: 当前类型
            
        Returns:
            选项列表
        """
        step_name = step_recipe.name
        step_params = step_recipe.params
        step_info = self.UTILITY_STEPS[step_name]
        new_type = step_info.get('output_type', current_type)
        
        # 从GremlinBase获取翻译
        desc = self.gremlin_base.get_token_desc(step_name)
        
        # math - 需要表达式字符串
        if step_info.get('needs_expression'):
            if step_params:
                expr = step_params[0]
            else:
                expr = '_ + 1'  # 默认表达式
            return [{
                'query_part': f".{step_name}('{expr}')",
                'desc_part': f"，{desc}",
                'new_label': current_label,
                'new_type': new_type
            }]
        
        # subgraph - 需要键参数
        elif step_info.get('needs_key'):
            if step_params:
                key = step_params[0]
            else:
                key = 'sg'  # 默认键
            return [{
                'query_part': f".{step_name}('{key}')",
                'desc_part': f"，{desc}",
                'new_label': current_label,
                'new_type': new_type
            }]
        
        # timeLimit - 需要数值参数
        elif step_info.get('needs_number'):
            if step_params:
                value = step_params[0]
            else:
                value = 1000  # 默认1秒
            return [{
                'query_part': f".{step_name}({value})",
                'desc_part': f"，{desc}",
                'new_label': current_label,
                'new_type': new_type
            }]
        
        # inject - 支持多参数
        elif step_info.get('multi_param'):
            if step_params:
                params_str = ', '.join([str(p) for p in step_params])
            else:
                params_str = '1, 2, 3'  # 默认值
            return [{
                'query_part': f".{step_name}({params_str})",
                'desc_part': f"，{desc}",
                'new_label': current_label,
                'new_type': new_type
            }]
        
        # call, io - 需要字符串参数
        elif step_info.get('needs_string'):
            if step_params:
                value = step_params[0]
            else:
                value = 'service' if step_name == 'call' else 'file.json'
            return [{
                'query_part': f".{step_name}('{value}')",
                'desc_part': f"，{desc}",
                'new_label': current_label,
                'new_type': new_type
            }]
        
        # 默认无参数
        return [{
            'query_part': f".{step_name}()",
            'desc_part': f"，{desc}",
            'new_label': current_label,
            'new_type': new_type
        }]
    
    #   M. 边修改步骤处理器  
    
    def _handle_edge_modification_step(self, step_recipe: Step, current_label: str,
                                      current_type: str) -> List[Dict]:
        """
        处理边修改步骤（from, to）
        
        这些步骤用于指定边的起点和终点
        
        Args:
            step_recipe: 配方步骤
            current_label: 当前标签
            current_type: 当前类型
            
        Returns:
            选项列表
        """
        from .GremlinExpr import AnonymousTraversal
        
        step_name = step_recipe.name
        step_params = step_recipe.params
        
        # 从GremlinBase获取翻译
        desc = self.gremlin_base.get_token_desc(step_name)
        
        if not step_params:
            # 无参数，生成默认标签
            label = 'a' if step_name == 'from' else 'b'
            return [{
                'query_part': f".{step_name}('{label}')",
                'desc_part': f"，{desc}",
                'new_label': current_label,
                'new_type': current_type
            }]
        
        param = step_params[0]
        
        # 如果是嵌套遍历
        if isinstance(param, AnonymousTraversal):
            nested_variants = self._generate_nested_traversal_variants(param, current_depth=0)
            result = []
            for variant in nested_variants[:1]:  # 只取第一个变体避免组合爆炸
                result.append({
                    'query_part': f".{step_name}({variant})",
                    'desc_part': f"，{desc}",
                    'new_label': current_label,
                    'new_type': current_type
                })
            return result if result else [{
                'query_part': f".{step_name}('a')",
                'desc_part': f"，{desc}",
                'new_label': current_label,
                'new_type': current_type
            }]
        
        # 字符串标签
        return [{
            'query_part': f".{step_name}('{param}')",
            'desc_part': f"，{desc}",
            'new_label': current_label,
            'new_type': current_type
        }]
    
    #   B. 属性访问步骤处理器  
    
    def _handle_property_access_step(self, step_recipe: Step, current_label: str,
                                    current_type: str) -> List[Dict]:
        """
        处理属性访问步骤（values, properties, valueMap等）
        
        支持两种模式：
        1. 无参数：返回所有属性
        2. 有参数：返回指定属性 + 泛化到同级属性
        
        Args:
            step_recipe: 配方步骤
            current_label: 当前标签
            current_type: 当前类型
            
        Returns:
            选项列表
        """
        step_name = step_recipe.name  
        step_params = step_recipe.params
        step_info = self.PROPERTY_ACCESS_STEPS[step_name]
        new_type = step_info['output_type']
        
        options = []
        
        # 如果没有当前标签，只能生成无参数版本
        if not current_label:
            step_desc = self.gremlin_base.get_token_desc(step_name)
            options.append({
                'query_part': f'.{step_name}()',
                'desc_part': f'，{step_desc}',
                'new_label': current_label,
                'new_type': new_type
            })
            return options
        
        # 获取当前标签的所有属性
        all_properties = [p['name'] for p in self.schema.get_properties_with_type(current_label)]
        
        if not all_properties:
            # 没有属性，返回无参数版本
            step_desc = self.gremlin_base.get_token_desc(step_name)
            options.append({
                'query_part': f'.{step_name}()',
                'desc_part': f'，{step_desc}',
                'new_label': current_label,
                'new_type': new_type
            })
            return options
        
        # 情况1：配方指定了属性
        if step_params and step_params[0]:
            # 支持多参数：values('name', 'age')
            recipe_props = step_params if isinstance(step_params, list) else [step_params]
            param_count = len(recipe_props)
            
            if self.controller:
                chain_category = self.controller.get_chain_category(len(self.recipe.steps))
                
                if param_count == 1:
                    # 单参数：使用原有的单参数泛化
                    selected_properties = self.controller.select_sibling_options(
                        recipe_option=recipe_props[0],
                        all_options=all_properties,
                        chain_category=chain_category
                    )
                    
                    for prop_name in selected_properties:
                        prop_desc = self.gremlin_base.get_schema_desc(prop_name)
                        step_desc = self.gremlin_base.get_token_desc(f'{step_name}_with_key')
                        if not step_desc or step_desc == f'{step_name}_with_key':
                            step_desc = self.gremlin_base.get_token_desc(step_name)
                        if '{}' in step_desc:
                            step_desc = step_desc.format(prop_desc)
                        else:
                            step_desc = f'{step_desc}：{prop_desc}'
                        options.append({
                            'query_part': f'.{step_name}("{prop_name}")',
                            'desc_part': f'，{step_desc}',
                            'new_label': current_label,
                            'new_type': new_type
                        })
                else:
                    # 多参数：使用多参数泛化，保持参数数量一致
                    prop_combinations = self.controller.select_multi_param_schema_options(
                        recipe_params=recipe_props,
                        all_options=all_properties,
                        chain_category=chain_category
                    )
                    
                    for combo in prop_combinations:
                        props_str = '", "'.join(combo)
                        props_desc = '、'.join([self.gremlin_base.get_schema_desc(p) for p in combo])
                        step_desc = self.gremlin_base.get_token_desc(f'{step_name}_with_key')
                        if not step_desc or step_desc == f'{step_name}_with_key':
                            step_desc = self.gremlin_base.get_token_desc(step_name)
                        if '{}' in step_desc:
                            step_desc = step_desc.format(props_desc)
                        else:
                            step_desc = f'{step_desc}：{props_desc}'
                        options.append({
                            'query_part': f'.{step_name}("{props_str}")',
                            'desc_part': f'，{step_desc}',
                            'new_label': current_label,
                            'new_type': new_type
                        })
            else:
                # 没有控制器，只保留原配方
                if param_count == 1:
                    prop_name = recipe_props[0]
                    if prop_name in all_properties:
                        prop_desc = self.gremlin_base.get_schema_desc(prop_name)
                        step_desc = self.gremlin_base.get_token_desc(f'{step_name}_with_key')
                        if not step_desc or step_desc == f'{step_name}_with_key':
                            step_desc = self.gremlin_base.get_token_desc(step_name)
                        if '{}' in step_desc:
                            step_desc = step_desc.format(prop_desc)
                        else:
                            step_desc = f'{step_desc}：{prop_desc}'
                        options.append({
                            'query_part': f'.{step_name}("{prop_name}")',
                            'desc_part': f'，{step_desc}',
                            'new_label': current_label,
                            'new_type': new_type
                        })
                else:
                    props_str = '", "'.join(recipe_props)
                    props_desc = '、'.join([self.gremlin_base.get_schema_desc(p) for p in recipe_props])
                    step_desc = self.gremlin_base.get_token_desc(f'{step_name}_with_key')
                    if not step_desc or step_desc == f'{step_name}_with_key':
                        step_desc = self.gremlin_base.get_token_desc(step_name)
                    if '{}' in step_desc:
                        step_desc = step_desc.format(props_desc)
                    else:
                        step_desc = f'{step_desc}：{props_desc}'
                    options.append({
                        'query_part': f'.{step_name}("{props_str}")',
                        'desc_part': f'，{step_desc}',
                        'new_label': current_label,
                        'new_type': new_type
                    })
        
        # 情况2：配方没有指定属性
        else:
            # 生成无参数版本
            step_desc = self.gremlin_base.get_token_desc(step_name)
            options.append({
                'query_part': f'.{step_name}()',
                'desc_part': f'，{step_desc}',
                'new_label': current_label,
                'new_type': new_type
            })
            
            # 可选：也生成一些带参数的版本（使用控制器限制数量）
            if self.controller and all_properties:
                chain_category = self.controller.get_chain_category(len(self.recipe.steps))
                strategy = self.controller.property_gen[chain_category]
                
                # 随机选择一些属性
                max_props = strategy.get('additional_random_max', 2)
                selected_count = min(max_props, len(all_properties))
                
                if selected_count > 0:
                    selected_props = random.sample(all_properties, selected_count)
                    for prop_name in selected_props:
                        prop_desc = self.gremlin_base.get_schema_desc(prop_name)
                        # 有参数时，使用 step_name_with_key 获取翻译
                        step_desc = self.gremlin_base.get_token_desc(f'{step_name}_with_key')
                        # 如果没有找到 _with_key 版本，回退到普通版本
                        if not step_desc or step_desc == f'{step_name}_with_key':
                            step_desc = self.gremlin_base.get_token_desc(step_name)
                        # 替换占位符
                        if '{}' in step_desc:
                            step_desc = step_desc.format(prop_desc)
                        else:
                            step_desc = f'{step_desc}：{prop_desc}'
                        options.append({
                            'query_part': f'.{step_name}("{prop_name}")',
                            'desc_part': f'，{step_desc}',
                            'new_label': current_label,
                            'new_type': new_type
                        })
        
        return options
    
    #   嵌套遍历泛化辅助方法  
    
    def _generate_nested_traversal_variants(self, anonymous_trav, current_depth=0):
        """
        递归生成嵌套遍历的变体
        
        Args:
            anonymous_trav: AnonymousTraversal 对象
            current_depth: 当前嵌套深度
            
        Returns:
            嵌套遍历字符串的列表
        """
        from .GremlinExpr import AnonymousTraversal
        
        # 获取配置
        max_depth = 3  # 默认值
        max_variants_per_step = 2  # 默认值
        
        if self.controller and hasattr(self.controller, 'config'):
            nested_config = self.controller.config.get('nested_traversal_strategy', {})
            max_depth = nested_config.get('max_nesting_depth', 3)
            max_variants_per_step = nested_config.get('max_variants_per_nested_step', 2)
        
        # 深度限制
        if current_depth >= max_depth:
            # 超过最大深度，返回占位符
            return ["..."]
        
        if not anonymous_trav or not anonymous_trav.steps:
            return [""]
        
        # 递归生成每个步骤的变体
        all_step_variants = []
        
        for step in anonymous_trav.steps:
            step_variants = []
            
            # 检查步骤是否包含嵌套遍历
            has_nested = False
            for param in step.params:
                if isinstance(param, AnonymousTraversal):
                    has_nested = True
                    break
            
            if has_nested:
                # 步骤包含嵌套遍历，递归处理
                for param_idx, param in enumerate(step.params):
                    if isinstance(param, AnonymousTraversal):
                        # 递归生成嵌套的变体
                        nested_variants = self._generate_nested_traversal_variants(
                            param, current_depth + 1
                        )
                        
                        # 为当前步骤生成变体（限制数量）
                        for nested_var in nested_variants[:max_variants_per_step]:
                            step_str = f"{step.name}({nested_var})"
                            step_variants.append(step_str)
                    else:
                        # 非嵌套参数，直接使用
                        pass
            else:
                # 步骤不包含嵌套遍历，生成简单变体
                step_variants = self._generate_simple_step_variants(
                    step, max_variants_per_step
                )
            
            all_step_variants.append(step_variants)
        
        # 组合所有步骤的变体
        if not all_step_variants:
            return [""]
        
        # 使用笛卡尔积组合（但限制总数）
        import itertools
        combinations = list(itertools.product(*all_step_variants))
        
        # 限制组合数量（避免爆炸）
        max_combinations = max_variants_per_step ** len(all_step_variants)
        max_combinations = min(max_combinations, 10)  # 最多10个组合
        
        if len(combinations) > max_combinations:
            import random
            combinations = random.sample(combinations, max_combinations)
        
        # 生成遍历字符串
        result = []
        for combo in combinations:
            traversal_str = ".".join(combo)
            result.append(traversal_str)
        
        return result
    
    def _generate_simple_step_variants(self, step, max_variants):
        """
        为简单步骤（不包含嵌套遍历）生成变体
        
        这个方法专门用于嵌套遍历内部的步骤泛化。
        策略：
        - Schema 属性（标签、边等）：固定泛化 max_variants 个
        - 数据值（具体的值）：只保留原值，不泛化
        
        Args:
            step: Step 对象
            max_variants: 最多生成的变体数（来自配置）
            
        Returns:
            步骤字符串的列表
        """
        variants = []
        step_name = step.name
        step_params = step.params
        
        # 根据步骤类型生成变体
        if step_name in ['out', 'in', 'both', 'outE', 'inE', 'bothE']:
            # 导航步骤：泛化边标签（Schema 属性）
            if step_params and isinstance(step_params[0], str):
                edge_label = step_params[0]
                all_edges = self.schema.get_edge_labels()
                
                # 原标签必须包含
                variants.append(f"{step_name}('{edge_label}')")
                
                # 添加其他边标签（限制数量）
                other_edges = [e for e in all_edges if e != edge_label]
                for edge in other_edges[:max_variants-1]:
                    variants.append(f"{step_name}('{edge}')")
            else:
                variants.append(f"{step_name}()")
                
        elif step_name == 'hasLabel':
            # hasLabel: 泛化顶点标签（Schema 属性）
            if len(step_params) == 1 and isinstance(step_params[0], str):
                label = step_params[0]
                all_labels = self.schema.get_vertex_labels()
                
                # 原标签必须包含
                variants.append(f"hasLabel('{label}')")
                
                # 添加其他标签（限制数量）
                other_labels = [l for l in all_labels if l != label]
                for other_label in other_labels[:max_variants-1]:
                    variants.append(f"hasLabel('{other_label}')")
            else:
                # 多参数或其他情况：保留原样
                params_str = ', '.join([self._format_param(p) for p in step_params])
                variants.append(f"hasLabel({params_str})")
                    
        elif step_name == 'has':
            # has: 区分 Schema 属性和数据值
            if len(step_params) == 2:
                # has('property', value) - 第一个是属性名（Schema），第二个是值（数据）
                # 嵌套内部：只保留原样，不泛化数据值
                param1_str = self._format_param(step_params[0])
                param2_str = self._format_param(step_params[1])
                variants.append(f"has({param1_str}, {param2_str})")
            elif len(step_params) == 1:
                # has('property') - 只有属性名
                param_str = self._format_param(step_params[0])
                variants.append(f"has({param_str})")
            else:
                variants.append(f"has(...)")
                
        elif step_name in ['hasKey', 'hasValue']:
            # hasKey/hasValue: 保留原样（数据值，不泛化）
            if len(step_params) == 1:
                param_str = self._format_param(step_params[0])
                variants.append(f"{step_name}({param_str})")
            else:
                params_str = ', '.join([self._format_param(p) for p in step_params])
                variants.append(f"{step_name}({params_str})")
                
        elif step_name in ['values', 'properties']:
            # 属性访问步骤：保留原样（属性名是 Schema，但在嵌套内部不泛化）
            if step_params:
                param_str = self._format_param(step_params[0])
                variants.append(f"{step_name}({param_str})")
            else:
                variants.append(f"{step_name}()")
                
        elif step_name in ['where', 'not']:
            # 嵌套的 where/not：保留原样（避免过深递归）
            variants.append(f"{step_name}(...)")
            
        else:
            # 其他步骤：保留原样
            if step_params:
                variants.append(f"{step_name}(...)")
            else:
                variants.append(f"{step_name}()")
        
        return variants[:max_variants]
    
    def _format_param(self, param):
        """格式化参数为字符串"""
        from .GremlinExpr import Predicate
        
        if isinstance(param, str):
            return f"'{param}'"
        elif isinstance(param, Predicate):
            return f"P.{param.operator}({param.value})"
        elif isinstance(param, (int, float)):
            return str(param)
        else:
            return "..."
    
    #   E. 过滤步骤处理器  
    
    def _handle_filter_step(self, step_recipe: Step, current_label: str,
                           current_type: str, remaining_steps: List[Step]) -> List[Dict]:
        """
        处理过滤步骤（hasLabel, has等）
        
        核心泛化逻辑：
        1. hasLabel: 泛化到所有同级顶点标签
        2. has: 泛化到所有同级属性 + 填充数据值
        
        Args:
            step_recipe: 配方步骤
            current_label: 当前标签
            current_type: 当前类型
            remaining_steps: 剩余步骤（用于判断是否是终端步骤）
            
        Returns:
            选项列表
        """
        step_name = step_recipe.name 
        step_params = step_recipe.params
        options = []
        
        # hasLabel() 步骤
        if step_name == 'hasLabel':
            if not step_params:
                return []
            
            # 支持多参数：hasLabel('person', 'movie')
            recipe_labels = step_params if isinstance(step_params, list) else [step_params]
            all_labels = self.schema.get_vertex_labels()
            param_count = len(recipe_labels)
            
            if self.controller:
                chain_category = self.controller.get_chain_category(len(self.recipe.steps))
                
                if param_count == 1:
                    # 单参数：使用原有的单参数泛化
                    selected_labels = self.controller.select_sibling_options(
                        recipe_option=recipe_labels[0],
                        all_options=all_labels,
                        chain_category=chain_category
                    )
                    
                    for label in selected_labels:
                        label_desc = self.gremlin_base.get_schema_desc(label)
                        options.append({
                            'query_part': f".hasLabel('{label}')",
                            'desc_part': f"，过滤出'{label_desc}'类型的顶点",
                            'new_label': label,
                            'new_type': current_type
                        })
                else:
                    # 多参数：使用多参数泛化，保持参数数量一致
                    label_combinations = self.controller.select_multi_param_schema_options(
                        recipe_params=recipe_labels,
                        all_options=all_labels,
                        chain_category=chain_category
                    )
                    
                    for combo in label_combinations:
                        labels_str = "', '".join(combo)
                        labels_desc = "、".join([self.gremlin_base.get_schema_desc(l) for l in combo])
                        options.append({
                            'query_part': f".hasLabel('{labels_str}')",
                            'desc_part': f"，过滤出'{labels_desc}'类型的顶点",
                            'new_label': combo[0],  # 使用第一个作为主标签
                            'new_type': current_type
                        })
            else:
                # 没有控制器，只保留原配方
                if param_count == 1:
                    label = recipe_labels[0]
                    if label in all_labels:
                        label_desc = self.gremlin_base.get_schema_desc(label)
                        options.append({
                            'query_part': f".hasLabel('{label}')",
                            'desc_part': f"，过滤出'{label_desc}'类型的顶点",
                            'new_label': label,
                            'new_type': current_type
                        })
                else:
                    labels_str = "', '".join(recipe_labels)
                    labels_desc = "、".join([self.gremlin_base.get_schema_desc(l) for l in recipe_labels])
                    options.append({
                        'query_part': f".hasLabel('{labels_str}')",
                        'desc_part': f"，过滤出'{labels_desc}'类型的顶点",
                        'new_label': recipe_labels[0],
                        'new_type': current_type
                    })
        
        # has() 步骤
        elif step_name == 'has':
            if not step_params:
                return []
            
            # has() 可以有多种形式：
            # 1. has('name') - 只有属性名
            # 2. has('name', 'Tom') - 属性名 + 单个值
            # 3. has('name', 'Tom', 'Jerry') - 属性名 + 多个值（OR语义）
            recipe_prop = step_params[0]
            recipe_values = step_params[1:] if len(step_params) > 1 else []
            value_count = len(recipe_values)
            
            # 判断是否是终端步骤
            is_terminal_step = (remaining_steps is None or len(remaining_steps) == 0)
            
            if current_label:
                # 知道当前标签，获取该标签的所有属性
                all_properties = [p['name'] for p in self.schema.get_properties_with_type(current_label)]
                
                # 使用控制器选择需要泛化的属性
                if self.controller:
                    chain_category = self.controller.get_chain_category(len(self.recipe.steps))
                    selected_properties = self.controller.select_sibling_options(
                        recipe_option=recipe_prop,
                        all_options=all_properties,
                        chain_category=chain_category
                    )
                else:
                    # 没有控制器，只使用配方指定的属性
                    selected_properties = [recipe_prop] if recipe_prop in all_properties else []
                
                # 为选中的属性生成选项
                for prop_name in selected_properties:
                    prop_info = next((p for p in self.schema.get_properties_with_type(current_label) 
                                    if p['name'] == prop_name), None)
                    if prop_info:
                        # 获取所有可用的值
                        all_values = self._get_multiple_values(current_label, prop_info)
                        
                        if value_count == 0:
                            # has('name') - 只有属性名，不需要值
                            prop_desc = self.gremlin_base.get_schema_desc(prop_name)
                            options.append({
                                'query_part': f".has('{prop_name}')",
                                'desc_part': f"，筛选包含'{prop_desc}'属性的元素",
                                'new_label': current_label,
                                'new_type': current_type
                            })
                        elif value_count == 1:
                            # has('name', 'Tom') - 单个值
                            if self.controller:
                                fill_count = self.controller.get_value_fill_count(
                                    is_terminal=is_terminal_step,
                                    available_count=len(all_values)
                                )
                            else:
                                fill_count = len(all_values)
                            
                            # 确保包含配方值
                            selected_values = [recipe_values[0]] if recipe_values[0] in all_values else []
                            other_values = [v for v in all_values if v != recipe_values[0]]
                            if other_values and fill_count > 1:
                                selected_values.extend(random.sample(other_values, min(fill_count - 1, len(other_values))))
                            
                            prop_desc = self.gremlin_base.get_schema_desc(prop_name)
                            for value in selected_values:
                                options.append({
                                    'query_part': f".has('{prop_name}', {repr(value)})",
                                    'desc_part': f"，其'{prop_desc}'为'{value}'",
                                    'new_label': current_label,
                                    'new_type': current_type
                                })
                        else:
                            # has('name', 'Tom', 'Jerry') - 多个值
                            if self.controller:
                                fill_times = self.controller.get_multi_param_value_fill_count(
                                    is_terminal=is_terminal_step
                                )
                            else:
                                fill_times = 1
                            
                            # 调整填充次数：不能超过实际可生成的不同组合数
                            if len(all_values) >= value_count:
                                # 使用集合去重，避免生成重复组合
                                generated_combos = set()
                                attempts = 0
                                max_attempts = fill_times * 10  # 避免无限循环
                                
                                while len(generated_combos) < fill_times and attempts < max_attempts:
                                    selected_combo = tuple(sorted(random.sample(all_values, value_count)))
                                    generated_combos.add(selected_combo)
                                    attempts += 1
                                
                                # 生成查询选项
                                prop_desc = self.gremlin_base.get_schema_desc(prop_name)
                                for combo in generated_combos:
                                    values_str = ", ".join(repr(v) for v in combo)
                                    options.append({
                                        'query_part': f".has('{prop_name}', {values_str})",
                                        'desc_part': f"，其'{prop_desc}'为{values_str}之一",
                                        'new_label': current_label,
                                        'new_type': current_type
                                    })
                            
                            # 确保包含原配方组合
                            values_str = ", ".join(repr(v) for v in recipe_values)
                            prop_desc = self.gremlin_base.get_schema_desc(prop_name)
                            recipe_option = {
                                'query_part': f".has('{prop_name}', {values_str})",
                                'desc_part': f"，其'{prop_desc}'为{values_str}之一",
                                'new_label': current_label,
                                'new_type': current_type
                            }
                            if recipe_option not in options:
                                options.insert(0, recipe_option)
            else:
                # 不知道当前标签，遍历所有标签
                for label in self.schema.get_vertex_labels():
                    prop_info = next((p for p in self.schema.get_properties_with_type(label) 
                                    if p['name'] == recipe_prop), None)
                    if prop_info:
                        all_values = self._get_multiple_values(label, prop_info)
                        
                        if value_count == 0:
                            # has('name') - 只有属性名
                            prop_desc = self.gremlin_base.get_schema_desc(recipe_prop)
                            options.append({
                                'query_part': f".has('{recipe_prop}')",
                                'desc_part': f"，筛选包含'{prop_desc}'属性的元素",
                                'new_label': label,
                                'new_type': current_type
                            })
                        elif value_count == 1:
                            # has('name', 'Tom') - 单个值
                            if self.controller:
                                fill_count = self.controller.get_value_fill_count(
                                    is_terminal=is_terminal_step,
                                    available_count=len(all_values)
                                )
                            else:
                                fill_count = len(all_values)
                            
                            selected_values = [recipe_values[0]] if recipe_values[0] in all_values else []
                            other_values = [v for v in all_values if v != recipe_values[0]]
                            if other_values and fill_count > 1:
                                selected_values.extend(random.sample(other_values, min(fill_count - 1, len(other_values))))
                            
                            prop_desc = self.gremlin_base.get_schema_desc(recipe_prop)
                            for value in selected_values:
                                options.append({
                                    'query_part': f".has('{recipe_prop}', {repr(value)})",
                                    'desc_part': f"，其'{prop_desc}'为'{value}'",
                                    'new_label': label,
                                    'new_type': current_type
                                })
                        else:
                            # has('name', 'Tom', 'Jerry') - 多个值
                            if self.controller:
                                fill_times = self.controller.get_multi_param_value_fill_count(
                                    is_terminal=is_terminal_step
                                )
                            else:
                                fill_times = 1
                            
                            # 调整填充次数：不能超过实际可生成的不同组合数
                            if len(all_values) >= value_count:
                                # 使用集合去重，避免生成重复组合
                                generated_combos = set()
                                attempts = 0
                                max_attempts = fill_times * 10  # 避免无限循环
                                
                                while len(generated_combos) < fill_times and attempts < max_attempts:
                                    selected_combo = tuple(sorted(random.sample(all_values, value_count)))
                                    generated_combos.add(selected_combo)
                                    attempts += 1
                                
                                # 生成查询选项
                                prop_desc = self.gremlin_base.get_schema_desc(recipe_prop)
                                for combo in generated_combos:
                                    values_str = ", ".join(repr(v) for v in combo)
                                    options.append({
                                        'query_part': f".has('{recipe_prop}', {values_str})",
                                        'desc_part': f"，其'{prop_desc}'为{values_str}之一",
                                        'new_label': label,
                                        'new_type': current_type
                                    })
                            
                            # 确保包含原配方组合
                            values_str = ", ".join(repr(v) for v in recipe_values)
                            prop_desc = self.gremlin_base.get_schema_desc(recipe_prop)
                            recipe_option = {
                                'query_part': f".has('{recipe_prop}', {values_str})",
                                'desc_part': f"，其'{prop_desc}'为{values_str}之一",
                                'new_label': label,
                                'new_type': current_type
                            }
                            if recipe_option not in options:
                                options.insert(0, recipe_option)
        
        # hasId() 步骤 - ID过滤
        elif step_name == 'hasId':
            if not step_params:
                return []
            
            # 支持多参数：hasId(1, 2, 3)
            recipe_ids = step_params if isinstance(step_params, list) else [step_params]
            all_ids = list(range(1, 101))
            param_count = len(recipe_ids)
            
            if self.controller:
                chain_category = self.controller.get_chain_category(len(self.recipe.steps))
                
                if param_count == 1:
                    # 单参数：使用原有的单参数泛化
                    recipe_id = recipe_ids[0]
                    strategy = self.controller.config['property_generalization'][chain_category]
                    max_ids = strategy.get('additional_random_max', 3)
                    selected_ids = [recipe_id] + random.sample([i for i in all_ids if i != recipe_id], 
                                                               min(max_ids, len(all_ids) - 1))
                    
                    for id_val in selected_ids:
                        options.append({
                            'query_part': f".hasId({id_val})",
                            'desc_part': f"，筛选ID为{id_val}的元素",
                            'new_label': current_label,
                            'new_type': current_type
                        })
                else:
                    # 多参数：使用多参数泛化，保持参数数量一致
                    id_combinations = self.controller.select_multi_param_schema_options(
                        recipe_params=[str(i) for i in recipe_ids],  # 转为字符串便于处理
                        all_options=[str(i) for i in all_ids],
                        chain_category=chain_category
                    )
                    
                    for combo in id_combinations:
                        # 转回整数
                        int_combo = [int(i) for i in combo]
                        ids_str = ", ".join(str(i) for i in int_combo)
                        options.append({
                            'query_part': f".hasId({ids_str})",
                            'desc_part': f"，筛选ID为{ids_str}的元素",
                            'new_label': current_label,
                            'new_type': current_type
                        })
            else:
                # 没有控制器，只保留原配方
                if param_count == 1:
                    options.append({
                        'query_part': f".hasId({recipe_ids[0]})",
                        'desc_part': f"，筛选ID为{recipe_ids[0]}的元素",
                        'new_label': current_label,
                        'new_type': current_type
                    })
                else:
                    ids_str = ", ".join(str(i) for i in recipe_ids)
                    options.append({
                        'query_part': f".hasId({ids_str})",
                        'desc_part': f"，筛选ID为{ids_str}的元素",
                        'new_label': current_label,
                        'new_type': current_type
                    })
        
        # is() 步骤 - 值比较（用于值流）
        elif step_name == 'is':
            if not step_params:
                return []
            
            recipe_value = step_params[0]
            
            # 判断是否是终端步骤
            is_terminal_step = (remaining_steps is None or len(remaining_steps) == 0)
            
            # 如果有当前标签，尝试从数据中获取值
            if current_label:
                # 获取该标签的所有属性
                all_properties = self.schema.get_properties_with_type(current_label)
                if all_properties:
                    # 随机选一个属性获取值
                    prop_info = random.choice(all_properties)
                    all_values = self._get_multiple_values(current_label, prop_info)
                    
                    # 使用控制器决定填充多少个值
                    if self.controller:
                        fill_count = self.controller.get_value_fill_count(
                            is_terminal=is_terminal_step,
                            available_count=len(all_values)
                        )
                    else:
                        fill_count = min(3, len(all_values))
                    
                    # 确保包含配方值
                    selected_values = [recipe_value]
                    other_values = [v for v in all_values if v != recipe_value]
                    if other_values and fill_count > 1:
                        selected_values.extend(random.sample(other_values, min(fill_count - 1, len(other_values))))
                else:
                    selected_values = [recipe_value]
            else:
                selected_values = [recipe_value]
            
            for value in selected_values:
                options.append({
                    'query_part': f".is({repr(value)})",
                    'desc_part': f"，判断值是否为{repr(value)}",
                    'new_label': current_label,
                    'new_type': current_type
                })
        
        # hasKey() 步骤 - 键过滤（用于属性流）
        elif step_name == 'hasKey':
            if not step_params:
                return []
            
            # 支持多参数：hasKey('name', 'age')
            recipe_keys = step_params if isinstance(step_params, list) else [step_params]
            
            # 如果有当前标签，获取该标签的所有属性键
            if current_label:
                all_keys = [p['name'] for p in self.schema.get_properties_with_type(current_label)]
            else:
                # 没有标签，获取所有可能的属性键
                all_keys = set()
                for label in self.schema.get_vertex_labels():
                    all_keys.update([p['name'] for p in self.schema.get_properties_with_type(label)])
                all_keys = list(all_keys)
            
            # 对每个配方键进行泛化
            all_selected_keys = set()
            for recipe_key in recipe_keys:
                if self.controller and all_keys:
                    chain_category = self.controller.get_chain_category(len(self.recipe.steps))
                    selected = self.controller.select_sibling_options(
                        recipe_option=recipe_key,
                        all_options=all_keys,
                        chain_category=chain_category
                    )
                    all_selected_keys.update(selected)
                else:
                    if recipe_key in all_keys:
                        all_selected_keys.add(recipe_key)
            
            # 生成单键选项
            for key in all_selected_keys:
                key_desc = self.gremlin_base.get_schema_desc(key)
                options.append({
                    'query_part': f".hasKey('{key}')",
                    'desc_part': f"，筛选包含键'{key_desc}'的属性",
                    'new_label': current_label,
                    'new_type': current_type
                })
            
            # 如果配方有多个键，也生成多键选项
            if len(recipe_keys) > 1:
                keys_str = "', '".join(recipe_keys)
                keys_desc = "、".join([self.gremlin_base.get_schema_desc(k) for k in recipe_keys])
                options.append({
                    'query_part': f".hasKey('{keys_str}')",
                    'desc_part': f"，筛选包含键'{keys_desc}'的属性",
                    'new_label': current_label,
                    'new_type': current_type
                })
        
        # hasValue() 步骤 - 值过滤（用于属性流）
        elif step_name == 'hasValue':
            if not step_params:
                return []
            
            # 支持多参数：hasValue('Tom', 'Jerry')
            recipe_values = step_params if isinstance(step_params, list) else [step_params]
            
            # 判断是否是终端步骤
            is_terminal_step = (remaining_steps is None or len(remaining_steps) == 0)
            
            # 如果有当前标签，从数据中获取值
            if current_label:
                all_properties = self.schema.get_properties_with_type(current_label)
                if all_properties:
                    # 从所有属性中收集值
                    all_values = []
                    for prop_info in all_properties:
                        values = self._get_multiple_values(current_label, prop_info)
                        all_values.extend(values)
                    
                    # 去重
                    all_values = list(set(all_values))
                    
                    # 使用控制器决定填充多少个值
                    if self.controller:
                        fill_count = self.controller.get_value_fill_count(
                            is_terminal=is_terminal_step,
                            available_count=len(all_values)
                        )
                    else:
                        fill_count = min(3, len(all_values))
                    
                    # 确保包含所有配方值
                    selected_values = list(recipe_values)
                    other_values = [v for v in all_values if v not in recipe_values]
                    if other_values and fill_count > len(recipe_values):
                        additional_count = min(fill_count - len(recipe_values), len(other_values))
                        selected_values.extend(random.sample(other_values, additional_count))
                else:
                    selected_values = recipe_values
            else:
                selected_values = recipe_values
            
            # 生成单值选项
            for value in selected_values:
                options.append({
                    'query_part': f".hasValue({repr(value)})",
                    'desc_part': f"，筛选包含值{repr(value)}的属性",
                    'new_label': current_label,
                    'new_type': current_type
                })
            
            # 如果配方有多个值，也生成多值选项
            if len(recipe_values) > 1:
                values_str = ", ".join(repr(v) for v in recipe_values)
                options.append({
                    'query_part': f".hasValue({values_str})",
                    'desc_part': f"，筛选包含值{values_str}的属性",
                    'new_label': current_label,
                    'new_type': current_type
                })
        
        # where() 步骤 - 条件过滤
        elif step_name == 'where':
            if not step_params:
                return []
            
            # where 步骤可以有多种形式：
            # 1. where(AnonymousTraversal) - 嵌套遍历
            # 2. where(Predicate) - 谓词
            # 3. where('property', Predicate) - 属性 + 谓词
            
            from .GremlinExpr import AnonymousTraversal, Predicate
            
            first_param = step_params[0]
            
            if isinstance(first_param, AnonymousTraversal):
                # 嵌套遍历：递归生成变体
                nested_variants = self._generate_nested_traversal_variants(first_param, current_depth=0)
                
                for nested_str in nested_variants:
                    options.append({
                        'query_part': f".where(__.{nested_str})",
                        'desc_part': "，条件过滤（嵌套遍历）",
                        'new_label': current_label,
                        'new_type': current_type
                    })
                        
            elif isinstance(first_param, Predicate):
                # 谓词：保留原谓词
                options.append({
                    'query_part': f".where(P.{first_param.operator}({first_param.value}))",
                    'desc_part': f"，条件过滤（{first_param.operator}）",
                    'new_label': current_label,
                    'new_type': current_type
                })
                        
            elif len(step_params) == 2 and isinstance(step_params[1], Predicate):
                # 属性 + 谓词：where('name', P.eq('Tom'))
                prop_name = step_params[0]
                predicate = step_params[1]
                
                prop_desc = self.gremlin_base.get_schema_desc(prop_name)
                options.append({
                    'query_part': f".where('{prop_name}', P.{predicate.operator}({predicate.value}))",
                    'desc_part': f"，条件过滤（{prop_desc} {predicate.operator}）",
                    'new_label': current_label,
                    'new_type': current_type
                })
            else:
                # 其他情况：保留原结构
                options.append({
                    'query_part': f".where(...)",
                    'desc_part': "，条件过滤",
                    'new_label': current_label,
                    'new_type': current_type
                })
            
            return options
        
        # not() 步骤 - 否定过滤
        elif step_name == 'not':
            if not step_params:
                return []
            
            # not 步骤通常接受一个 AnonymousTraversal 作为参数
            # 例如：not(__.out('knows'))
            
            from .GremlinExpr import AnonymousTraversal
            
            first_param = step_params[0]
            
            if isinstance(first_param, AnonymousTraversal):
                # 嵌套遍历：递归生成变体
                nested_variants = self._generate_nested_traversal_variants(first_param, current_depth=0)
                
                for nested_str in nested_variants:
                    options.append({
                        'query_part': f".not(__.{nested_str})",
                        'desc_part': "，否定过滤（嵌套遍历）",
                        'new_label': current_label,
                        'new_type': current_type
                    })
            else:
                # 其他类型的参数：保留原结构
                options.append({
                    'query_part': f".not(...)",
                    'desc_part': "，否定过滤",
                    'new_label': current_label,
                    'new_type': current_type
                })
        
        # filter() 步骤 - 通用过滤
        elif step_name == 'filter':
            if not step_params:
                return []
            
            from .GremlinExpr import AnonymousTraversal
            
            first_param = step_params[0]
            
            if isinstance(first_param, AnonymousTraversal):
                # 嵌套遍历：递归生成变体
                nested_variants = self._generate_nested_traversal_variants(first_param, current_depth=0)
                
                for nested_str in nested_variants[:1]:  # 只取第一个避免组合爆炸
                    options.append({
                        'query_part': f".filter({nested_str})",
                        'desc_part': "，通用过滤",
                        'new_label': current_label,
                        'new_type': current_type
                    })
            else:
                options.append({
                    'query_part': f".filter(...)",
                    'desc_part': "，通用过滤",
                    'new_label': current_label,
                    'new_type': current_type
                })
        
        # and() 步骤 - 逻辑与
        elif step_name == 'and':
            if not step_params:
                return []
            
            from .GremlinExpr import AnonymousTraversal
            
            # and接受多个遍历作为参数
            if all(isinstance(p, AnonymousTraversal) for p in step_params):
                # 为每个参数生成一个变体（避免组合爆炸）
                variants_list = []
                for param in step_params:
                    variants = self._generate_nested_traversal_variants(param, current_depth=0)
                    variants_list.append(variants[0] if variants else "__.identity()")
                
                variants_str = ", ".join(variants_list)
                options.append({
                    'query_part': f".and({variants_str})",
                    'desc_part': "，逻辑与过滤",
                    'new_label': current_label,
                    'new_type': current_type
                })
            else:
                options.append({
                    'query_part': f".and(...)",
                    'desc_part': "，逻辑与过滤",
                    'new_label': current_label,
                    'new_type': current_type
                })
        
        # or() 步骤 - 逻辑或
        elif step_name == 'or':
            if not step_params:
                return []
            
            from .GremlinExpr import AnonymousTraversal
            
            # or接受多个遍历作为参数
            if all(isinstance(p, AnonymousTraversal) for p in step_params):
                # 为每个参数生成一个变体（避免组合爆炸）
                variants_list = []
                for param in step_params:
                    variants = self._generate_nested_traversal_variants(param, current_depth=0)
                    variants_list.append(variants[0] if variants else "__.identity()")
                
                variants_str = ", ".join(variants_list)
                options.append({
                    'query_part': f".or({variants_str})",
                    'desc_part': "，逻辑或过滤",
                    'new_label': current_label,
                    'new_type': current_type
                })
            else:
                options.append({
                    'query_part': f".or(...)",
                    'desc_part': "，逻辑或过滤",
                    'new_label': current_label,
                    'new_type': current_type
                })
            
            return options
        
        # as() 步骤 - 标记步骤
        elif step_name == 'as':
            if not step_params:
                return []
            
            # as('label') - 为当前步骤添加标记
            label_name = step_params[0]
            options.append({
                'query_part': f".as('{label_name}')",
                'desc_part': f"，标记为'{label_name}'",
                'new_label': current_label,  # 保持当前标签不变
                'new_type': current_type  # 保持当前类型不变
            })
            
            return options
        
        return options
    
    #   D. 导航步骤处理器  
    
    def _handle_navigation_step(self, step_recipe: Step, current_label: str,
                               current_type: str) -> List[Dict]:
        """
        处理导航步骤（out, in, both, outE, inE, bothE等）
        
        核心泛化逻辑：
        - 泛化到所有同级边标签
        
        Args:
            step_recipe: 配方步骤
            current_label: 当前标签
            current_type: 当前类型
            
        Returns:
            选项列表
        """
        step_name = step_recipe.name 
        step_params = step_recipe.params
        options = []
        
        # 确定导航方向和目标类型
        if step_name in ['out', 'in', 'both']:
            # 顶点导航
            target_type = 'vertex'
        elif step_name in ['outE', 'inE', 'bothE']:
            # 边导航
            target_type = 'edge'
        elif step_name in ['outV', 'inV', 'otherV']:
            # 从边到顶点
            target_type = 'vertex'
        else:
            return []
        
        # 如果没有当前标签，无法确定可用的边
        if not current_label:
            # 生成无参数版本
            options.append({
                'query_part': f'.{step_name}()',
                'desc_part': f'，沿{step_name}方向遍历',
                'new_label': None,
                'new_type': target_type
            })
            return options
        
        # 获取当前标签可用的边
        if step_name in ['out', 'outE']:
            # 出边：source == current_label
            all_edges = [{'label': label, 'target_label': edge_info['destination']} 
                        for label, edge_info in self.schema.edges.items() 
                        if edge_info['source'] == current_label]
        elif step_name in ['in', 'inE']:
            # 入边：destination == current_label
            all_edges = [{'label': label, 'target_label': edge_info['source']} 
                        for label, edge_info in self.schema.edges.items() 
                        if edge_info['destination'] == current_label]
        elif step_name in ['both', 'bothE']:
            # 双向边
            outgoing = [{'label': label, 'target_label': edge_info['destination']} 
                       for label, edge_info in self.schema.edges.items() 
                       if edge_info['source'] == current_label]
            incoming = [{'label': label, 'target_label': edge_info['source']} 
                       for label, edge_info in self.schema.edges.items() 
                       if edge_info['destination'] == current_label]
            all_edges = outgoing + incoming
        else:
            # outV, inV, otherV 不需要边标签
            options.append({
                'query_part': f'.{step_name}()',
                'desc_part': f'，获取{step_name}顶点',
                'new_label': None,
                'new_type': target_type
            })
            return options
        
        if not all_edges:
            # 没有可用的边，返回无参数版本
            options.append({
                'query_part': f'.{step_name}()',
                'desc_part': f'，沿{step_name}方向遍历',
                'new_label': None,
                'new_type': target_type
            })
            return options
        
        # 提取边标签
        edge_labels = list(set([edge['label'] for edge in all_edges]))
        
        # 如果配方指定了边标签
        if step_params and step_params[0]:
            # 支持多参数：out('knows', 'created')
            recipe_edges = step_params if isinstance(step_params, list) else [step_params]
            param_count = len(recipe_edges)
            
            # 使用控制器选择需要泛化的边标签
            if self.controller:
                chain_category = self.controller.get_chain_category(len(self.recipe.steps))
                
                if param_count == 1:
                    # 单参数：使用原有的单参数泛化
                    selected_edges = self.controller.select_sibling_options(
                        recipe_option=recipe_edges[0],
                        all_options=edge_labels,
                        chain_category=chain_category
                    )
                else:
                    # 多参数：使用多参数泛化，保持参数数量一致
                    edge_combinations = self.controller.select_multi_param_schema_options(
                        recipe_params=recipe_edges,
                        all_options=edge_labels,
                        chain_category=chain_category
                    )
                    # 对于多参数，我们需要生成组合形式
                    selected_edges = edge_combinations
            else:
                # 没有控制器，只使用配方指定的边
                if param_count == 1:
                    selected_edges = [recipe_edges[0]] if recipe_edges[0] in edge_labels else []
                else:
                    # 多参数情况，保留原配方组合
                    selected_edges = [recipe_edges]
        else:
            # 配方没有指定边标签，使用控制器选择一些
            if self.controller:
                chain_category = self.controller.get_chain_category(len(self.recipe.steps))
                strategy = self.controller.property_gen[chain_category]
                max_edges = strategy.get('additional_random_max', 2)
                selected_count = min(max_edges, len(edge_labels))
                selected_edges = random.sample(edge_labels, selected_count) if selected_count > 0 else []
            else:
                selected_edges = edge_labels[:2]  # 默认选择前2个
        
        # 为选中的边生成选项
        for edge_item in selected_edges:
            # 判断是单个边标签还是边标签组合
            if isinstance(edge_item, list):
                # 多参数组合
                edges_str = "', '".join(edge_item)
                edges_desc = '、'.join([self.gremlin_base.get_schema_desc(e) for e in edge_item])
                
                # 对于多参数，目标标签不确定（可能有多个），设为 None
                options.append({
                    'query_part': f".{step_name}('{edges_str}')",
                    'desc_part': f"，沿'{edges_desc}'边{step_name}方向遍历",
                    'new_label': None,
                    'new_type': target_type
                })
            else:
                # 单个边标签
                edge_label = edge_item
                # 找到对应的边信息，确定目标标签
                edge_info = next((e for e in all_edges if e['label'] == edge_label), None)
                if edge_info:
                    target_label = edge_info.get('target_label')
                    edge_desc = self.gremlin_base.get_schema_desc(edge_label)
                    
                    # 根据目标类型设置 new_label
                    # - 如果目标是顶点，new_label 是目标顶点的标签
                    # - 如果目标是边，new_label 是边的标签（用于访问边的属性）
                    if target_type == 'vertex':
                        new_label = target_label
                    elif target_type == 'edge':
                        new_label = edge_label  # 边的标签，用于后续访问边的属性
                    else:
                        new_label = None
                    
                    options.append({
                        'query_part': f".{step_name}('{edge_label}')",
                        'desc_part': f"，沿'{edge_desc}'边{step_name}方向遍历",
                        'new_label': new_label,
                        'new_type': target_type
                    })
        
        # 如果没有生成任何选项，添加无参数版本
        if not options:
            options.append({
                'query_part': f'.{step_name}()',
                'desc_part': f'，沿{step_name}方向遍历',
                'new_label': None,
                'new_type': target_type
            })
        
        return options
    
    #   J. 特殊步骤处理器  
    
    def _handle_special_step(self, step_recipe: Step, current_label: str,
                            current_type: str, remaining_steps: List[Step]) -> List[Dict]:
        """
        处理特殊步骤
        
        已实现：
        - V(): 起始步骤
        - addV(): 添加顶点
        
        未实现（按需添加）：
        - addE(): 添加边
        - property(): 设置属性
        - choose/coalesce/optional: 分支逻辑
        - repeat/until/times/emit: 循环逻辑
        - match: 模式匹配
        - select/project: 投影
        - path/tree: 路径
        - union/flatMap/map: 高阶操作
        
        Args:
            step_recipe: 配方步骤
            current_label: 当前标签
            current_type: 当前类型
            remaining_steps: 剩余步骤
            
        Returns:
            选项列表
        """
        step_name = step_recipe.name 
        step_params = step_recipe.params
        
        # V() 步骤
        if step_name == 'V':
            if step_params:
                # V(id) - 指定ID
                ids = ', '.join([repr(p) for p in step_params])
                return [{
                    'query_part': f".V({ids})",
                    'desc_part': f"查找ID为{ids}的顶点",
                    'new_label': None,
                    'new_type': 'vertex'
                }]
            else:
                # V() - 所有顶点
                return [{
                    'query_part': ".V()",
                    'desc_part': "查找所有顶点",
                    'new_label': None,
                    'new_type': 'vertex'
                }]
        
        # addV() 步骤
        elif step_name == 'addV':
            if not step_params:
                return []
            
            label = step_params[0]
            creation_info = self.schema.get_vertex_creation_info(label)
            
            query_part = f".addV('{label}')"
            desc_part = f"添加一个'{self.gremlin_base.get_schema_desc(label)}'顶点"
            
            # 添加必需属性
            for prop_name in creation_info.get('required', []):
                prop_info = next((p for p in self.schema.get_properties_with_type(label) 
                                if p['name'] == prop_name), None)
                if prop_info:
                    prop_value = self._get_random_value(label, prop_info, for_update=True)
                    query_part += f".property('{prop_name}', {repr(prop_value)})"
                    desc_part += f"，并设置其'{self.gremlin_base.get_schema_desc(prop_name)}'为'{prop_value}'"
            
            return [{
                'query_part': query_part,
                'desc_part': desc_part,
                'new_label': label,
                'new_type': 'vertex'
            }]
        
        # E() 步骤 - 边遍历起点
        elif step_name == 'E':
            if step_params:
                # E(id1, id2, ...) - 指定ID的边，支持多参数
                param_count = len(step_params)
                all_ids = list(range(1, 101))
                
                if self.controller and param_count > 1:
                    # 多参数：使用多参数泛化
                    chain_category = self.controller.get_chain_category(len(self.recipe.steps))
                    id_combinations = self.controller.select_multi_param_schema_options(
                        recipe_params=[str(i) for i in step_params],
                        all_options=[str(i) for i in all_ids],
                        chain_category=chain_category
                    )
                    
                    options = []
                    for combo in id_combinations:
                        int_combo = [int(i) for i in combo]
                        ids_str = ", ".join(str(i) for i in int_combo)
                        options.append({
                            'query_part': f".E({ids_str})",
                            'desc_part': f"查找ID为{ids_str}的边",
                            'new_label': None,
                            'new_type': 'edge'
                        })
                    return options
                else:
                    # 单参数或无控制器：保留原配方
                    ids = ', '.join([repr(p) for p in step_params])
                    return [{
                        'query_part': f".E({ids})",
                        'desc_part': f"查找ID为{ids}的边",
                        'new_label': None,
                        'new_type': 'edge'
                    }]
            else:
                # E() - 所有边
                return [{
                    'query_part': ".E()",
                    'desc_part': "查找所有边",
                    'new_label': None,
                    'new_type': 'edge'
                }]
        
        # select() 步骤 - 投影选择
        elif step_name == 'select':
            if not step_params:
                return []
            
            from .GremlinExpr import AnonymousTraversal
            
            desc = self.gremlin_base.get_token_desc('select')
            
            # 检查第一个参数是否是嵌套遍历
            first_param = step_params[0]
            if isinstance(first_param, AnonymousTraversal):
                # select(__.out()) - 嵌套遍历
                nested_variants = self._generate_nested_traversal_variants(first_param, current_depth=0)
                
                options = []
                for nested_str in nested_variants:
                    options.append({
                        'query_part': f".select(__.{nested_str})",
                        'desc_part': f"，{desc}（嵌套遍历）",
                        'new_label': None,
                        'new_type': 'value'
                    })
                return options
            
            elif len(step_params) == 1:
                # select('a') - 选择单个标记
                param = step_params[0]
                return [{
                    'query_part': f".select('{param}')",
                    'desc_part': f"，{desc}标记'{param}'",
                    'new_label': None,
                    'new_type': 'value'
                }]
            else:
                # select('a', 'b', ...) - 选择多个标记
                params_str = ', '.join([f"'{p}'" for p in step_params])
                return [{
                    'query_part': f".select({params_str})",
                    'desc_part': f"，{desc}多个标记",
                    'new_label': None,
                    'new_type': 'map'
                }]
        
        # path() 步骤 - 获取遍历路径
        elif step_name == 'path':
            desc = self.gremlin_base.get_token_desc('path')
            return [{
                'query_part': ".path()",
                'desc_part': f"，{desc}",
                'new_label': None,
                'new_type': 'path'
            }]
        
        # addE() 步骤 - 添加边
        elif step_name == 'addE':
            from .GremlinExpr import AnonymousTraversal
            if not step_params:
                return []
            
            # 检查参数类型
            if isinstance(step_params[0], str):
                # string变体: addE('knows')
                label = step_params[0]
                return [{
                    'query_part': f".addE('{label}')",
                    'desc_part': f"添加标签为'{label}'的边",
                    'new_label': label,
                    'new_type': 'edge'
                }]
            elif isinstance(step_params[0], AnonymousTraversal):
                # traversal变体: addE(__.constant('knows'))
                # 生成嵌套遍历的变体
                variants = self._generate_nested_traversal_variants(step_params[0], current_depth=0)
                result = []
                for variant in variants[:2]:  # 最多2个变体
                    result.append({
                        'query_part': f".addE(__.{variant})",
                        'desc_part': "添加边（通过遍历获取标签）",
                        'new_label': None,
                        'new_type': 'edge'
                    })
                return result if result else []
            else:
                return []
        
        # property() 步骤 - 设置属性
        elif step_name == 'property':
            if not step_params:
                return []
            
            # 检查不同的property变体
            if len(step_params) == 1:
                # property(map) 变体
                map_param = step_params[0]
                return [{
                    'query_part': f".property({map_param})",
                    'desc_part': "设置属性（映射）",
                    'new_label': current_label,
                    'new_type': current_type
                }]
            
            elif len(step_params) >= 2:
                # 检查第一个参数是否是cardinality
                first_param = step_params[0]
                if isinstance(first_param, dict) and 'type' in first_param:
                    # property(cardinality, key, value, ...) 变体
                    card_type = first_param['type']
                    card_value = first_param.get('value', 'multi')
                    
                    if len(step_params) == 2:
                        # property(cardinality, map)
                        map_param = step_params[1]
                        return [{
                            'query_part': f".property({card_type}('{card_value}'), {map_param})",
                            'desc_part': f"设置属性（{card_type}基数，映射）",
                            'new_label': current_label,
                            'new_type': current_type
                        }]
                    else:
                        # property(cardinality, key, value, ...)
                        key = step_params[1]
                        value = step_params[2]
                        extra_params = step_params[3:] if len(step_params) > 3 else []
                        
                        result = []
                        # 基础变体
                        if extra_params:
                            extra_str = ", ".join([f"'{p}'" if isinstance(p, str) else str(p) for p in extra_params])
                            query_part = f".property({card_type}('{card_value}'), '{key}', '{value}', {extra_str})"
                        else:
                            query_part = f".property({card_type}('{card_value}'), '{key}', '{value}')"
                        
                        result.append({
                            'query_part': query_part,
                            'desc_part': f"设置属性 {key}={value}（{card_type}基数）",
                            'new_label': current_label,
                            'new_type': current_type
                        })
                        return result
                else:
                    # property(key, value, ...) 变体（无cardinality）
                    key = step_params[0]
                    value = step_params[1]
                    extra_params = step_params[2:] if len(step_params) > 2 else []
                    
                    result = []
                    # 基础变体
                    if extra_params:
                        extra_str = ", ".join([f"'{p}'" if isinstance(p, str) else str(p) for p in extra_params])
                        query_part = f".property('{key}', '{value}', {extra_str})"
                    else:
                        query_part = f".property('{key}', '{value}')"
                    
                    result.append({
                        'query_part': query_part,
                        'desc_part': f"设置属性 {key}={value}",
                        'new_label': current_label,
                        'new_type': current_type
                    })
                    return result
            
            return []
        
        # project() 步骤 - 投影
        elif step_name == 'project':
            if not step_params:
                return []
            keys_str = "', '".join(step_params)
            return [{
                'query_part': f".project('{keys_str}')",
                'desc_part': f"投影字段",
                'new_label': None,
                'new_type': 'map'
            }]
        
        # union() 步骤 - 联合
        elif step_name == 'union':
            from .GremlinExpr import AnonymousTraversal
            if not step_params or not all(isinstance(p, AnonymousTraversal) for p in step_params):
                return []
            
            # 为每个嵌套遍历生成变体（每个参数独立泛化）
            all_param_variants = []
            for param in step_params:
                variants = self._generate_nested_traversal_variants(param, current_depth=0)
                if variants:
                    # 每个参数最多取2个变体（根据嵌套策略）
                    all_param_variants.append([f"__.{v}" for v in variants[:2]])
                else:
                    all_param_variants.append(["__.identity()"])  # 默认值
            
            # 组合所有参数的变体（笛卡尔积）
            import itertools
            result = []
            combinations = list(itertools.product(*all_param_variants))
            
            # 限制组合数量（避免爆炸）
            max_combinations = min(len(combinations), 4)  # 最多4个组合
            if len(combinations) > max_combinations:
                import random
                combinations = random.sample(combinations, max_combinations)
            
            for combo in combinations:
                union_str = ", ".join(combo)
                result.append({
                    'query_part': f".union({union_str})",
                    'desc_part': "联合多个遍历",
                    'new_label': None,
                    'new_type': current_type
                })
            
            return result if result else []
        
        # coalesce() 步骤 - 合并
        elif step_name == 'coalesce':
            from .GremlinExpr import AnonymousTraversal
            if not step_params or not all(isinstance(p, AnonymousTraversal) for p in step_params):
                return []
            
            # 为每个嵌套遍历生成变体（每个参数独立泛化）
            all_param_variants = []
            for param in step_params:
                variants = self._generate_nested_traversal_variants(param, current_depth=0)
                if variants:
                    # 每个参数最多取2个变体（根据嵌套策略）
                    all_param_variants.append([f"__.{v}" for v in variants[:2]])
                else:
                    all_param_variants.append(["__.identity()"])  # 默认值
            
            # 组合所有参数的变体（笛卡尔积）
            import itertools
            result = []
            combinations = list(itertools.product(*all_param_variants))
            
            # 限制组合数量（避免爆炸）
            max_combinations = min(len(combinations), 4)  # 最多4个组合
            if len(combinations) > max_combinations:
                import random
                combinations = random.sample(combinations, max_combinations)
            
            for combo in combinations:
                coalesce_str = ", ".join(combo)
                result.append({
                    'query_part': f".coalesce({coalesce_str})",
                    'desc_part': "合并遍历（返回第一个非空结果）",
                    'new_label': None,
                    'new_type': current_type
                })
            
            return result if result else []
        
        # choose() 步骤 - 条件分支
        elif step_name == 'choose':
            from .GremlinExpr import AnonymousTraversal, Predicate
            if not step_params:
                return []
            
            # choose有多个变体，根据参数数量和类型处理
            if len(step_params) == 1:
                # choose(traversal) - 单参数
                if isinstance(step_params[0], AnonymousTraversal):
                    variants = self._generate_nested_traversal_variants(step_params[0], current_depth=0)
                    result = []
                    for variant in variants[:2]:
                        result.append({
                            'query_part': f".choose(__.{variant})",
                            'desc_part': "条件分支",
                            'new_label': None,
                            'new_type': current_type
                        })
                    return result if result else []
            
            elif len(step_params) == 2:
                # choose(traversal, traversal) - 两参数
                if all(isinstance(p, AnonymousTraversal) for p in step_params):
                    # 为每个参数生成变体
                    variants1 = self._generate_nested_traversal_variants(step_params[0], current_depth=0)
                    variants2 = self._generate_nested_traversal_variants(step_params[1], current_depth=0)
                    
                    result = []
                    # 组合（限制数量）
                    for v1 in variants1[:2]:
                        for v2 in variants2[:2]:
                            result.append({
                                'query_part': f".choose(__.{v1}, __.{v2})",
                                'desc_part': "条件分支（if-then）",
                                'new_label': None,
                                'new_type': current_type
                            })
                            if len(result) >= 4:  # 最多4个组合
                                break
                        if len(result) >= 4:
                            break
                    return result if result else []
            
            elif len(step_params) == 3:
                # choose(traversal, traversal, traversal) - 三参数
                if all(isinstance(p, AnonymousTraversal) for p in step_params):
                    # 为每个参数生成变体
                    variants1 = self._generate_nested_traversal_variants(step_params[0], current_depth=0)
                    variants2 = self._generate_nested_traversal_variants(step_params[1], current_depth=0)
                    variants3 = self._generate_nested_traversal_variants(step_params[2], current_depth=0)
                    
                    result = []
                    # 组合（限制数量）
                    for v1 in variants1[:2]:
                        for v2 in variants2[:1]:  # 减少组合
                            for v3 in variants3[:1]:
                                result.append({
                                    'query_part': f".choose(__.{v1}, __.{v2}, __.{v3})",
                                    'desc_part': "条件分支（if-then-else）",
                                    'new_label': None,
                                    'new_type': current_type
                                })
                                if len(result) >= 4:  # 最多4个组合
                                    break
                            if len(result) >= 4:
                                break
                        if len(result) >= 4:
                            break
                    return result if result else []
            
            return []
        
        # optional() 步骤 - 可选遍历
        elif step_name == 'optional':
            from .GremlinExpr import AnonymousTraversal
            if not step_params or not isinstance(step_params[0], AnonymousTraversal):
                return []
            
            nested_variants = self._generate_nested_traversal_variants(step_params[0], current_depth=0)
            if nested_variants:
                return [{
                    'query_part': f".optional(__.{nested_variants[0]})",
                    'desc_part': "可选遍历",
                    'new_label': None,
                    'new_type': current_type
                }]
            return []
        
        # repeat() 步骤 - 重复遍历
        elif step_name == 'repeat':
            from .GremlinExpr import AnonymousTraversal
            if not step_params:
                return []
            
            # 检查是否是string+traversal变体
            if len(step_params) == 2 and isinstance(step_params[0], str) and isinstance(step_params[1], AnonymousTraversal):
                # repeat(label, traversal)
                label = step_params[0]
                nested_variants = self._generate_nested_traversal_variants(step_params[1], current_depth=0)
                result = []
                for variant in nested_variants[:2]:
                    result.append({
                        'query_part': f".repeat('{label}', __.{variant})",
                        'desc_part': f"重复遍历（标签：{label}）",
                        'new_label': None,
                        'new_type': current_type
                    })
                return result if result else []
            
            # 单参数traversal变体
            elif len(step_params) == 1 and isinstance(step_params[0], AnonymousTraversal):
                nested_variants = self._generate_nested_traversal_variants(step_params[0], current_depth=0)
                result = []
                for variant in nested_variants[:2]:
                    result.append({
                        'query_part': f".repeat(__.{variant})",
                        'desc_part': "重复遍历",
                        'new_label': None,
                        'new_type': current_type
                    })
                return result if result else []
            
            return []
        
        # until() 步骤 - 终止条件
        elif step_name == 'until':
            from .GremlinExpr import AnonymousTraversal, Predicate
            if not step_params:
                return []
            
            # 检查参数类型
            if isinstance(step_params[0], AnonymousTraversal):
                # traversal变体
                nested_variants = self._generate_nested_traversal_variants(step_params[0], current_depth=0)
                result = []
                for variant in nested_variants[:2]:
                    result.append({
                        'query_part': f".until(__.{variant})",
                        'desc_part': "终止条件（遍历）",
                        'new_label': None,
                        'new_type': current_type
                    })
                return result if result else []
            elif isinstance(step_params[0], Predicate):
                # predicate变体
                pred_str = self._format_predicate(step_params[0])
                return [{
                    'query_part': f".until({pred_str})",
                    'desc_part': "终止条件（谓词）",
                    'new_label': None,
                    'new_type': current_type
                }]
            
            return []
        
        # times() 步骤 - 重复次数
        elif step_name == 'times':
            if not step_params:
                return []
            times_count = step_params[0]
            return [{
                'query_part': f".times({times_count})",
                'desc_part': f"重复{times_count}次",
                'new_label': current_label,
                'new_type': current_type
            }]
        
        # emit() 步骤 - 发射中间结果
        elif step_name == 'emit':
            from .GremlinExpr import AnonymousTraversal, Predicate
            
            if not step_params:
                # emit() 无参数
                return [{
                    'query_part': ".emit()",
                    'desc_part': "发射中间结果",
                    'new_label': current_label,
                    'new_type': current_type
                }]
            
            # 检查参数类型
            if isinstance(step_params[0], AnonymousTraversal):
                # emit(traversal) 带遍历条件
                nested_variants = self._generate_nested_traversal_variants(step_params[0], current_depth=0)
                result = []
                for variant in nested_variants[:2]:
                    result.append({
                        'query_part': f".emit(__.{variant})",
                        'desc_part': "条件发射（遍历）",
                        'new_label': current_label,
                        'new_type': current_type
                    })
                return result if result else []
            elif isinstance(step_params[0], Predicate):
                # emit(predicate) 带谓词条件
                pred_str = self._format_predicate(step_params[0])
                return [{
                    'query_part': f".emit({pred_str})",
                    'desc_part': "条件发射（谓词）",
                    'new_label': current_label,
                    'new_type': current_type
                }]
            
            return []
        
        # flatMap() 步骤 - 扁平映射
        elif step_name == 'flatMap':
            from .GremlinExpr import AnonymousTraversal
            if not step_params or not isinstance(step_params[0], AnonymousTraversal):
                return []
            
            nested_variants = self._generate_nested_traversal_variants(step_params[0], current_depth=0)
            if nested_variants:
                return [{
                    'query_part': f".flatMap(__.{nested_variants[0]})",
                    'desc_part': "扁平映射",
                    'new_label': None,
                    'new_type': current_type
                }]
            return []
        
        # map() 步骤 - 映射
        elif step_name == 'map':
            from .GremlinExpr import AnonymousTraversal
            if not step_params or not isinstance(step_params[0], AnonymousTraversal):
                return []
            
            nested_variants = self._generate_nested_traversal_variants(step_params[0], current_depth=0)
            if nested_variants:
                return [{
                    'query_part': f".map(__.{nested_variants[0]})",
                    'desc_part': "映射",
                    'new_label': None,
                    'new_type': 'value'
                }]
            return []
        
        # match() 步骤 - 模式匹配
        elif step_name == 'match':
            from .GremlinExpr import AnonymousTraversal
            if not step_params or not all(isinstance(p, AnonymousTraversal) for p in step_params):
                return []
            
            # 为每个模式生成变体（每个参数独立泛化）
            all_pattern_variants = []
            for param in step_params:
                variants = self._generate_nested_traversal_variants(param, current_depth=0)
                if variants:
                    # 每个模式最多取2个变体（根据嵌套策略）
                    all_pattern_variants.append([f"__.{v}" for v in variants[:2]])
                else:
                    all_pattern_variants.append(["__.identity()"])  # 默认值
            
            # 组合所有模式的变体（笛卡尔积）
            import itertools
            result = []
            combinations = list(itertools.product(*all_pattern_variants))
            
            # 限制组合数量（避免爆炸）
            max_combinations = min(len(combinations), 4)  # 最多4个组合
            if len(combinations) > max_combinations:
                import random
                combinations = random.sample(combinations, max_combinations)
            
            for combo in combinations:
                match_str = ", ".join(combo)
                result.append({
                    'query_part': f".match({match_str})",
                    'desc_part': "模式匹配",
                    'new_label': None,
                    'new_type': 'map'
                })
            
            return result if result else []
        
        # tree() 步骤 - 树结构
        elif step_name == 'tree':
            return [{
                'query_part': ".tree()",
                'desc_part': "构建树结构",
                'new_label': None,
                'new_type': 'tree'
            }]
        
        # 其他特殊步骤暂未实现
        return []
    
    def _apply_random_enhancement(self, query: str, desc: str, 
                                 current_label: str, current_type: str) -> List[Tuple[str, str]]:
        """
        应用随机增强（可选功能）
        
        根据流类型智能选择增强步骤：
        - 元素流（vertex, edge）：limit, range, sample, dedup, order
        - 值流（value, string, number）：limit, dedup, order
        
        Args:
            query: 当前查询字符串
            desc: 当前描述字符串
            current_label: 当前标签
            current_type: 当前类型
            
        Returns:
            增强后的查询-描述对列表
        """
        results = []
        
        # 判断流类型
        is_element_stream = current_type in ['vertex', 'edge', 'element']
        is_value_stream = current_type in ['value', 'string', 'number', 'list', 'map']
        
        # 如果是终端类型（boolean, none）或不适合增强的类型，跳过增强
        if current_type in ['boolean', 'none', 'graph']:
            return results
        
        # 检查查询中已存在的步骤，避免重复添加
        existing_steps = set()
        import re
        # 提取查询中的所有步骤名（不带参数）
        step_pattern = r'\.(\w+)\('
        for match in re.finditer(step_pattern, query):
            existing_steps.add(match.group(1))
        
        # 定义可用的增强步骤及其权重
        enhancements = []
        
        # === 元素流增强选项 ===
        if is_element_stream:
            # limit - 40% 权重
            if random.random() < 0.4:
                # 70% 使用常见值，30% 使用随机值
                if random.random() < 0.7:
                    limit_value = random.choice([1, 3, 5, 10, 20, 50, 100])
                else:
                    limit_value = random.randint(1, 200)
                
                limit_desc = self.gremlin_base.get_token_desc('limit')
                if '{}' in limit_desc:
                    limit_desc = limit_desc.format(limit_value)
                enhancements.append({
                    'query_part': f'.limit({limit_value})',
                    'desc_part': f'，{limit_desc}',
                    'new_type': current_type,
                    'weight': 0.4
                })
            
            # range - 20% 权重
            if random.random() < 0.2:
                # 60% 使用常见范围，40% 使用随机范围
                if random.random() < 0.6:
                    start, end = random.choice([(0, 5), (1, 10), (5, 20), (0, 10)])
                else:
                    start = random.randint(0, 50)
                    end = start + random.randint(10, 100)
                
                range_desc = self.gremlin_base.get_token_desc('range')
                if not range_desc or range_desc == 'range':
                    range_desc = '取范围内的结果'
                enhancements.append({
                    'query_part': f'.range({start}, {end})',
                    'desc_part': f'，{range_desc}',
                    'new_type': current_type,
                    'weight': 0.2
                })
            
            # sample - 30% 权重
            if random.random() < 0.3:
                # 80% 使用常见值，20% 使用随机值
                if random.random() < 0.8:
                    sample_value = random.choice([1, 2, 3, 5, 10])
                else:
                    sample_value = random.randint(1, 50)
                
                sample_desc = self.gremlin_base.get_token_desc('sample')
                if '{}' in sample_desc:
                    sample_desc = sample_desc.format(sample_value)
                elif not sample_desc or sample_desc == 'sample':
                    sample_desc = f'随机采样 {sample_value} 个结果'
                enhancements.append({
                    'query_part': f'.sample({sample_value})',
                    'desc_part': f'，{sample_desc}',
                    'new_type': current_type,
                    'weight': 0.3
                })
            
            # dedup - 30% 权重（检查是否已存在）
            if random.random() < 0.3 and 'dedup' not in existing_steps:
                dedup_desc = self.gremlin_base.get_token_desc('dedup')
                enhancements.append({
                    'query_part': '.dedup()',
                    'desc_part': f'，{dedup_desc}',
                    'new_type': current_type,
                    'weight': 0.3
                })
            
            # order - 20% 权重（检查是否已存在）
            if random.random() < 0.2 and 'order' not in existing_steps:
                order_desc = self.gremlin_base.get_token_desc('order')
                enhancements.append({
                    'query_part': '.order()',
                    'desc_part': f'，{order_desc}',
                    'new_type': current_type,
                    'weight': 0.2
                })
        
        # === 值流增强选项 ===
        elif is_value_stream:
            # limit - 40% 权重
            if random.random() < 0.4:
                # 70% 使用常见值，30% 使用随机值
                if random.random() < 0.7:
                    limit_value = random.choice([1, 3, 5, 10, 20, 50])
                else:
                    limit_value = random.randint(1, 100)
                
                limit_desc = self.gremlin_base.get_token_desc('limit')
                if '{}' in limit_desc:
                    limit_desc = limit_desc.format(limit_value)
                enhancements.append({
                    'query_part': f'.limit({limit_value})',
                    'desc_part': f'，{limit_desc}',
                    'new_type': current_type,
                    'weight': 0.4
                })
            
            # dedup - 40% 权重（值流中去重更重要，检查是否已存在）
            if random.random() < 0.4 and 'dedup' not in existing_steps:
                dedup_desc = self.gremlin_base.get_token_desc('dedup')
                enhancements.append({
                    'query_part': '.dedup()',
                    'desc_part': f'，{dedup_desc}',
                    'new_type': current_type,
                    'weight': 0.4
                })
            
            # order - 30% 权重（值流中排序较常见，检查是否已存在）
            if random.random() < 0.3 and 'order' not in existing_steps:
                order_desc = self.gremlin_base.get_token_desc('order')
                enhancements.append({
                    'query_part': '.order()',
                    'desc_part': f'，{order_desc}',
                    'new_type': current_type,
                    'weight': 0.3
                })
        
        # 随机选择一个增强步骤
        if enhancements:
            enhancement = random.choice(enhancements)
            enhanced_query = query + enhancement['query_part']
            enhanced_desc = desc + enhancement['desc_part']
            results.append((enhanced_query, enhanced_desc))
        
        return results
    
    #   辅助方法  
    
    def _get_random_value(self, label: str, prop_info: Dict, for_update: bool = False) -> Any:
        """根据属性类型生成随机值"""
        prop_name, prop_type = prop_info['name'], prop_info['type']
        instance = self.schema.get_instance(label)
        
        if instance and prop_name in instance and not for_update:
            value = instance.get(prop_name)
            if value is not None:
                return value
        
        if prop_type == 'STRING':
            return ''.join(random.choices(string.ascii_letters, k=random.randint(5, 8)))
        if prop_type in ['INT32', 'INT64']:
            return random.randint(1, 10000)
        
        return "default_value"
    
    def _get_multiple_values(self, label: str, prop_info: Dict, 
                            for_update: bool = False) -> List[Any]:
        """获取多个真实数据值"""
        prop_name, prop_type = prop_info['name'], prop_info['type']
        instances = self.schema.get_instances(label)
        
        values = []
        for instance in instances:
            if instance and prop_name in instance and not for_update:
                value = instance.get(prop_name)
                if value is not None:
                    values.append(value)
        
        # 如果没有真实数据，生成随机值
        if not values:
            if prop_type == 'STRING':
                values = [''.join(random.choices(string.ascii_letters, k=random.randint(5, 8))) 
                         for _ in range(random.randint(2, 5))]
            elif prop_type in ['INT32', 'INT64']:
                values = [random.randint(1, 10000) for _ in range(random.randint(2, 5))]
            else:
                values = ["default_value"]
        
        return values
