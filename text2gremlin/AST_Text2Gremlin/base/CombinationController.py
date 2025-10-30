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
组合爆炸控制器

提供统一的配置驱动的控制策略，适用于所有Gremlin步骤和谓词的泛化生成。
"""

import random
from typing import List, Dict, Any


class CombinationController:
    """组合爆炸控制器 - 基于配置文件的统一控制策略"""
    
    def __init__(self, config: Dict):
        """
        初始化控制器
        
        Args:
            config: 从combination_control_config.json加载的配置字典
        """
        self.config = config
        
        # 验证必要配置项并加载
        try:
            # 链长度分类阈值
            self.chain_thresholds = config['chain_thresholds']
            
            # 随机增强控制
            self.random_enhancement = config['random_enhancement']
            
            # 数据填充策略
            self.value_fill = config['value_fill_strategy']
            
            # 属性泛化策略
            self.property_gen = config['property_generalization']
        except KeyError as e:
            raise ValueError(f"缺少必要配置项: {e}") from None
        
        # 总数限制（可选）
        self.max_total = config.get('max_total_combinations', {})
        
        # 验证关键类别的存在性
        # chain_thresholds 只需要 short, medium, long（ultra 通过 else 分支隐式定义）
        for category in ('short', 'medium', 'long'):
            if category not in self.chain_thresholds:
                raise ValueError(f"chain_thresholds 缺少 '{category}' 配置")
        
        # property_generalization 需要所有4个类别（包括 ultra）
        for category in ('short', 'medium', 'long', 'ultra'):
            if category not in self.property_gen:
                raise ValueError(f"property_generalization 缺少 '{category}' 配置")
            # 验证每个类别的必要字段
            required_fields = ['full_coverage_threshold', 'additional_random_min', 'additional_random_max']
            for field in required_fields:
                if field not in self.property_gen[category]:
                    raise ValueError(f"property_generalization.{category} 缺少 '{field}' 字段")
        
    def get_chain_category(self, step_count: int) -> str:
        """
        根据步骤数确定链长度类别
        
        Args:
            step_count: 查询步骤数量
            
        Returns:
            'short' | 'medium' | 'long' | 'ultra'
        """
        if step_count <= self.chain_thresholds['short']:
            return 'short'
        elif step_count <= self.chain_thresholds['medium']:
            return 'medium'
        elif step_count <= self.chain_thresholds['long']:
            return 'long'
        else:
            return 'ultra'
    
    def should_apply_random_enhancement(self, is_terminal: bool, 
                                       enhancement_count: int) -> bool:
        """
        判断是否应该应用随机增强
        
        Args:
            is_terminal: 是否是终端步骤
            enhancement_count: 当前查询已经应用的增强次数
            
        Returns:
            True表示应该应用随机增强
        """
        # 检查是否超过最大增强次数
        if enhancement_count >= self.random_enhancement['max_enhancements_per_query']:
            return False
        
        # 根据位置决定概率
        probability = (self.random_enhancement['terminal_step_probability'] 
                      if is_terminal 
                      else self.random_enhancement['middle_step_probability'])
        
        return random.random() < probability
    
    def get_value_fill_count(self, is_terminal: bool, available_count: int) -> int:
        """
        决定应该填充多少个数据值
        
        Args:
            is_terminal: 是否是终端步骤（配方的最后一步）
            available_count: CSV中实际可用的数据数量
            
        Returns:
            应该填充的值数量
        """
        if not is_terminal:
            # 中间步骤：固定1个
            return min(self.value_fill['middle_step']['count'], available_count)
        else:
            # 终端步骤：随机2-3个
            min_count = self.value_fill['terminal_step']['min']
            max_count = self.value_fill['terminal_step']['max']
            target = random.randint(min_count, max_count)
            return min(target, available_count)
    
    def select_sibling_options(self, recipe_option: str, all_options: List[str],
                               chain_category: str) -> List[str]:
        """
        通用的同级选项选择器 - 核心泛化方法
        
        适用于所有需要选择同级选项的场景：
        - 顶点标签: hasLabel('person') → 其他顶点标签
        - 边标签: out('acted_in') → 其他边标签
        - 顶点属性: has('name', ?) → 其他顶点属性
        - 边属性: has('role', ?) → 其他边属性
        - 以及任何具有平级关系的选项
        
        Args:
            recipe_option: 配方中指定的选项（必须包含）
            all_options: 所有同级选项列表
            chain_category: 链长度类别 ('short'|'medium'|'long'|'ultra')
            
        Returns:
            选中的选项列表（包含recipe_option + 随机选择的其他选项）
        """
        strategy = self.property_gen[chain_category]
        
        # 1. 配方选项必须包含
        if recipe_option and recipe_option in all_options:
            selected = [recipe_option]
            other_options = [opt for opt in all_options if opt != recipe_option]
        else:
            # 如果配方选项不在列表中，从头开始
            selected = []
            other_options = all_options
        
        # 2. 判断是否全部遍历
        if len(all_options) <= strategy['full_coverage_threshold']:
            # 同级选项少，全部遍历
            return list(all_options)
        
        # 3. 同级选项多，随机选择额外的
        additional_count = random.randint(
            strategy['additional_random_min'],
            strategy['additional_random_max']
        )
        
        if additional_count > 0 and other_options:
            # 随机采样，确保不重复
            sample_count = min(additional_count, len(other_options))
            selected.extend(random.sample(other_options, sample_count))
        
        return selected
    
    def select_multi_param_schema_options(self, recipe_params: List[str], 
                                         all_options: List[str],
                                         chain_category: str) -> List[List[str]]:
        """
        多参数Schema填充选择器
        
        适用于从schema获取参数的多参数步骤：
        - hasLabel('person', 'movie') → 其他2参数标签组合
        - hasKey('name', 'age') → 其他2参数属性键组合
        - out('acted_in', 'directed') → 其他2参数边标签组合
        
        Args:
            recipe_params: 配方中的参数列表 ['person', 'movie']
            all_options: 所有可选项 ['person', 'movie', 'user', 'genre', 'keyword']
            chain_category: 链长度类别
            
        Returns:
            多参数组合列表 [['person', 'movie'], ['user', 'genre'], ...]
        """
        param_count = len(recipe_params)
        if param_count <= 1:
            # 单参数情况，使用原有方法
            return [self.select_sibling_options(recipe_params[0] if recipe_params else '', 
                                               all_options, chain_category)]
        
        # 获取多参数策略配置
        multi_config = self.config.get('multi_param_strategy', {})
        schema_config = multi_config.get('schema_fill', {})
        max_combinations = schema_config.get(chain_category, {}).get('max_combinations', 1)
        
        combinations = []
        seen = set()  # 用于去重的集合
        
        # 1. 保留原配方组合
        combinations.append(recipe_params.copy())
        seen.add(tuple(sorted(recipe_params)))
        
        if max_combinations <= 1:
            return combinations
        
        # 2. 生成其他同参数数量的组合
        other_options = [opt for opt in all_options if opt not in recipe_params]
        
        # 情况1：可选参数个数 < 多参数个数
        # 例如：需要3个参数，但只有1个可选项
        # 策略：不生成额外组合（因为无法组成完整的多参数组合）
        if len(other_options) < param_count:
            return combinations  # 只返回原配方
        
        # 情况2：可选参数个数 = 多参数个数
        # 例如：需要2个参数，恰好有2个可选项
        # 策略：不生成额外组合（因为只有一种组合方式，随机选择没意义）
        if len(other_options) == param_count:
            return combinations  # 只返回原配方
        
        # 情况3：可选参数个数 > 多参数个数
        # 例如：需要3个参数，有5个可选项
        # 策略：随机生成多个组合
        attempts = 0
        max_attempts = 20  # 避免无限循环
        
        while len(combinations) < max_combinations and attempts < max_attempts:
            # 随机选择同数量的参数
            combo = random.sample(other_options, param_count)
            
            # 使用排序后的元组作为key进行去重（因为参数顺序不影响语义）
            key = tuple(sorted(combo))
            if key not in seen:
                seen.add(key)
                combinations.append(combo)
            
            attempts += 1
        
        return combinations
    
    def get_multi_param_value_fill_count(self, is_terminal: bool) -> int:
        """
        多参数数据值填充次数控制
        
        Args:
            is_terminal: 是否是终端步骤
            
        Returns:
            填充次数（每次填充的值个数由调用方根据参数个数决定）
        """
        multi_config = self.config.get('multi_param_strategy', {})
        value_config = multi_config.get('value_fill', {})
        
        if not is_terminal:
            # 中间步骤：只填充1次
            return value_config.get('middle_step', {}).get('fill_times', 1)
        else:
            # 终端步骤：填充2-3次
            min_times = value_config.get('terminal_step', {}).get('fill_times_min', 2)
            max_times = value_config.get('terminal_step', {}).get('fill_times_max', 3)
            return random.randint(min_times, max_times)
    
    def should_stop_generation(self, current_count: int, 
                               chain_category: str) -> bool:
        """
        判断是否应该停止生成（可选功能）
        
        Args:
            current_count: 当前已生成的查询数量
            chain_category: 链长度类别
            
        Returns:
            True表示应该停止生成
        """
        max_limit = self.max_total.get(chain_category)
        if max_limit is None:
            return False
        
        return current_count >= max_limit
    
    def print_strategy_info(self, step_count: int):
        """
        打印当前查询的控制策略信息（用于调试）
        
        Args:
            step_count: 查询步骤数量
        """
        category = self.get_chain_category(step_count)
        strategy = self.property_gen[category]
        
        print(f"🎯 组合控制策略")
        print(f"  步骤数: {step_count}")
        print(f"  类别: {category}")
        print(f"  总数限制: {self.max_total.get(category, '无')}")
        print(f"  随机增强: 中间{self.random_enhancement['middle_step_probability']*100:.0f}%, "
              f"末尾{self.random_enhancement['terminal_step_probability']*100:.0f}%, "
              f"最多{self.random_enhancement['max_enhancements_per_query']}次")
        print(f"  数据填充: 中间{self.value_fill['middle_step']['count']}个, "
              f"终端{self.value_fill['terminal_step']['min']}-{self.value_fill['terminal_step']['max']}个")
        print(f"  属性泛化: 阈值≤{strategy['full_coverage_threshold']}全遍历, "
              f"否则配方+随机{strategy['additional_random_min']}-{strategy['additional_random_max']}个")


# 使用示例
if __name__ == '__main__':
    import json
    
    # 加载配置
    with open('combination_control_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建控制器
    controller = CombinationController(config)
    
    # 测试不同长度的链
    print("=" * 60)
    for steps in [3, 5, 7, 10]:
        controller.print_strategy_info(steps)
        print()
    
    # 测试同级选项选择
    print("=" * 60)
    print("🧪 测试同级选项选择:")
    
    test_cases = [
        {
            'name': '5个顶点标签',
            'recipe': 'person',
            'all': ['person', 'movie', 'genre', 'keyword', 'user'],
            'category': 'short'
        },
        {
            'name': '10个属性',
            'recipe': 'name',
            'all': ['name', 'born', 'bio', 'poster', 'height', 'weight', 
                   'age', 'gender', 'nationality', 'occupation'],
            'category': 'medium'
        }
    ]
    
    for case in test_cases:
        print(f"\n{case['name']} ({case['category']}链):")
        selected = controller.select_sibling_options(
            case['recipe'], case['all'], case['category']
        )
        print(f"  配方: {case['recipe']}")
        print(f"  总数: {len(case['all'])}")
        print(f"  选中: {selected} ({len(selected)}个)")
