"""
Gremlin查询生成器核心引擎。

基于递归回溯算法，从结构化配方生成大量多样化的Gremlin查询及其中文描述。
"""

import os
import random
import string
from typing import List, Dict, Any, Tuple, Set

# 导入我们定义好的核心数据结构和 Schema
from Schema import Schema
from GremlinParse import Traversal, Step
from GremlinExpr import Predicate
from GremlinBase import GremlinBase
from Config import Config

class TraversalGenerator:
    def __init__(self, schema: Schema, recipe: Traversal, gremlin_base: GremlinBase):
        self.schema = schema
        self.recipe = recipe
        self.gremlin_base = gremlin_base
        self.generated_pairs: Set[Tuple[str, str]] = set()

    def generate(self) -> List[Tuple[str, str]]:
        """
        主生成函数，负责启动递归生成过程。

        Returns:
            一个去重后的 (查询, 描述) 对的列表。
        """
        self.generated_pairs.clear()
        self._recursive_generate(
            recipe_steps=self.recipe.steps,
            current_query="g",
            current_desc="从图中开始",
            current_label=None,
            current_type='graph'
        )
        return list(self.generated_pairs)

    def _get_random_value(self, label: str, prop_info: Dict, for_update: bool = False) -> Any:
        """根据属性类型，智能地生成一个随机值。"""
        prop_name, prop_type = prop_info['name'], prop_info['type']
        instance = self.schema.get_instance(label)
        if instance and prop_name in instance and not for_update:
            value = instance.get(prop_name)
            if value is not None: return value
        if prop_type == 'STRING':
            return ''.join(random.choices(string.ascii_letters, k=random.randint(5, 8)))
        if prop_type in ['INT32', 'INT64']:
            return random.randint(1, 10000)
        return "default_value"
    
    def _get_multiple_values(self, label: str, prop_info: Dict, for_update: bool = False) -> List[Any]:
        """获取多个真实数据值，用于生成多个查询变体。"""
        prop_name, prop_type = prop_info['name'], prop_info['type']
        instances = self.schema.get_instances(label)
        
        values = []
        for instance in instances:
            if instance and prop_name in instance and not for_update:
                value = instance.get(prop_name)
                if value is not None:
                    values.append(value)
        
        # 如果没有获取到真实数据，生成一些随机值
        if not values:
            if prop_type == 'STRING':
                values = [''.join(random.choices(string.ascii_letters, k=random.randint(5, 8))) for _ in range(random.randint(2, 5))]
            elif prop_type in ['INT32', 'INT64']:
                values = [random.randint(1, 10000) for _ in range(random.randint(2, 5))]
            else:
                values = ["default_value"]
        
        return values

    def _recursive_generate(self, recipe_steps: List[Step], current_query: str, current_desc: str, current_label: str, current_type: str):
        """【核心】递归生成函数，实现深度优先搜索和回溯。"""
        if not recipe_steps:
            # 当到达配方末端时，尝试随机增强（20%概率）
            if random.random() < 0.2:
                enhanced_queries = self._apply_random_enhancements(current_query, current_desc, current_label, current_type)
                for enhanced_query, enhanced_desc in enhanced_queries:
                    self.generated_pairs.add((enhanced_query, enhanced_desc))
            return

        step_recipe = recipe_steps[0]
        remaining_steps = recipe_steps[1:]
        
        # 获取当前状态下，该步骤的所有合法“填充”选项
        options = self._get_valid_options_for_step(step_recipe, current_label, current_type)
        
        # 遍历每一个合法选项，继续向下探索
        for option in options:
            next_query = current_query + option['query_part']
            next_desc = current_desc + option['desc_part']
            
            # 【保存中间结果】
            self.generated_pairs.add((next_query, next_desc))

            if option['new_type'] != 'none':
                # 继续递归
                self._recursive_generate(remaining_steps, next_query, next_desc, option['new_label'], option['new_type'])
                
                # 在中间步骤也有小概率（10%）进行随机增强
                if remaining_steps and random.random() < 0.1:
                    enhanced_queries = self._apply_random_enhancements(next_query, next_desc, option['new_label'], option['new_type'])
                    for enhanced_query, enhanced_desc in enhanced_queries:
                        # 对增强后的查询继续执行剩余步骤
                        self._recursive_generate(remaining_steps, enhanced_query, enhanced_desc, option['new_label'], option['new_type'])

    def _get_valid_options_for_step(self, step_recipe: Step, current_label: str, current_type: str) -> List[Dict]:
        """根据配方中的一步，返回所有合法的实例化选项。"""
        step_name = step_recipe.name.lower()
        step_params = step_recipe.params
        options = []

        # --- 起始步骤 ---
        if current_type == 'graph':
            if step_name == 'v':
                # V() 步骤只是开始遍历，不指定具体的标签
                # 标签过滤由后续的 hasLabel 步骤处理
                if step_params:
                    # 如果 V() 有参数（ID），直接使用
                    ids = ', '.join([repr(p) for p in step_params])
                    options.append({
                        'query_part': f".V({ids})",
                        'desc_part': f"查找ID为{ids}的顶点",
                        'new_label': None, 'new_type': 'vertex'
                    })
                else:
                    # 无参数的 V()，返回所有顶点
                    options.append({
                        'query_part': ".V()",
                        'desc_part': "查找所有顶点",
                        'new_label': None, 'new_type': 'vertex'
                    })
            elif step_name == 'addv':
                label = step_params[0]
                creation_info = self.schema.get_vertex_creation_info(label)
                query_part = f".addV('{label}')"
                desc_part = f"添加一个'{self.gremlin_base.get_schema_desc(label)}'顶点"
                for prop_name in creation_info.get('required', []):
                    prop_info = next(p for p in self.schema.get_properties_with_type(label) if p['name'] == prop_name)
                    prop_value = self._get_random_value(label, prop_info, for_update=True)
                    query_part += f".property('{prop_name}', {repr(prop_value)})"
                    desc_part += f"，并设置其'{self.gremlin_base.get_schema_desc(prop_name)}'为'{prop_value}'"
                options.append({'query_part': query_part, 'desc_part': desc_part, 'new_label': label, 'new_type': 'vertex'})
            return options

        # --- 后续步骤 ---
        if step_name in ['out', 'in', 'both']:
            if current_label:  # 只有当我们知道当前标签时才能导航
                valid_steps = self.schema.get_valid_steps(current_label, current_type)
                possible_edges = next((s['params'] for s in valid_steps if s['step'] == step_name), [])
                
                if step_params and step_params[0] in possible_edges:
                    # 优先使用配方中指定的边
                    edge = step_params[0]
                    new_label, new_type = self.schema.get_step_result_label(current_label, {'step': step_name, 'param': edge})
                    desc_map = {'out': '出边', 'in': '入边', 'both': '双向边'}
                    options.append({
                        'query_part': f".{step_name}('{edge}')",
                        'desc_part': f"，然后沿着'{self.gremlin_base.get_schema_desc(edge)}'的{desc_map[step_name]}找到'{self.gremlin_base.get_schema_desc(new_label)}'顶点",
                        'new_label': new_label, 'new_type': new_type
                    })
                    
                    # 同时也生成其他可能的边变体（用于泛化）
                    for other_edge in possible_edges:
                        if other_edge != edge:  # 避免重复
                            new_label, new_type = self.schema.get_step_result_label(current_label, {'step': step_name, 'param': other_edge})
                            options.append({
                                'query_part': f".{step_name}('{other_edge}')",
                                'desc_part': f"，然后沿着'{self.gremlin_base.get_schema_desc(other_edge)}'的{desc_map[step_name]}找到'{self.gremlin_base.get_schema_desc(new_label)}'顶点",
                                'new_label': new_label, 'new_type': new_type
                            })
                else:
                    # 如果没有指定边或指定的边无效，尝试所有可能的边
                    for edge in possible_edges:
                        new_label, new_type = self.schema.get_step_result_label(current_label, {'step': step_name, 'param': edge})
                        desc_map = {'out': '出边', 'in': '入边', 'both': '双向边'}
                        options.append({
                            'query_part': f".{step_name}('{edge}')",
                            'desc_part': f"，然后沿着'{self.gremlin_base.get_schema_desc(edge)}'的{desc_map[step_name]}找到'{self.gremlin_base.get_schema_desc(new_label)}'顶点",
                            'new_label': new_label, 'new_type': new_type
                        })

        elif step_name == 'has':
            prop_name = step_params[0]
            
            if current_label:
                # 如果知道当前标签，使用该标签的多个真实数据值
                prop_info = next((p for p in self.schema.get_properties_with_type(current_label) if p['name'] == prop_name), None)
                if prop_info:
                    # 获取多个真实数据值
                    values = self._get_multiple_values(current_label, prop_info)
                    for value in values:
                        options.append({
                            'query_part': f".has('{prop_name}', {repr(value)})",
                            'desc_part': f"，其'{self.gremlin_base.get_schema_desc(prop_name)}'属性为'{value}'",
                            'new_label': current_label, 'new_type': current_type
                        })
            else:
                # 如果不知道当前标签，尝试所有有该属性的标签
                for label in self.schema.get_vertex_labels():
                    prop_info = next((p for p in self.schema.get_properties_with_type(label) if p['name'] == prop_name), None)
                    if prop_info:
                        # 获取多个真实数据值
                        values = self._get_multiple_values(label, prop_info)
                        for value in values:
                            options.append({
                                'query_part': f".has('{prop_name}', {repr(value)})",
                                'desc_part': f"，其'{self.gremlin_base.get_schema_desc(prop_name)}'属性为'{value}'",
                                'new_label': label, 'new_type': current_type
                            })
        
        elif step_name == 'property': # 更新
            prop_name = step_params[0]
            prop_info = next((p for p in self.schema.get_updatable_properties(current_label) if p['name'] == prop_name), None)
            if prop_info:
                # 对于更新操作，我们可以使用多个不同的值来生成多个变体
                values = self._get_multiple_values(current_label, prop_info, for_update=True)
                for value in values:
                    options.append({
                        'query_part': f".property('{prop_name}', {repr(value)})",
                        'desc_part': f"，并将其'{self.gremlin_base.get_schema_desc(prop_name)}'属性更新为'{value}'",
                        'new_label': current_label, 'new_type': current_type
                    })

        elif step_name == 'limit':
            num = step_params[0] if step_params else random.randint(1, 10)
            options.append({
                'query_part': f".limit({num})",
                'desc_part': f"，并只取前{num}个结果",
                'new_label': current_label, 'new_type': current_type
            })

        elif step_name == 'haslabel':
            # 处理 hasLabel 步骤
            if step_params:
                # 如果配方中指定了标签，优先使用指定的标签
                target_label = step_params[0]
                if target_label in self.schema.get_vertex_labels():
                    options.append({
                        'query_part': f".hasLabel('{target_label}')",
                        'desc_part': f"，过滤出'{self.gremlin_base.get_schema_desc(target_label)}'类型的顶点",
                        'new_label': target_label, 'new_type': current_type
                    })
                
                # 同时也生成其他可能的标签变体（用于泛化）
                for label in self.schema.get_vertex_labels():
                    if label != target_label:  # 避免重复
                        options.append({
                            'query_part': f".hasLabel('{label}')",
                            'desc_part': f"，过滤出'{self.gremlin_base.get_schema_desc(label)}'类型的顶点",
                            'new_label': label, 'new_type': current_type
                        })
            else:
                # 如果没有指定标签，尝试所有可能的标签
                for label in self.schema.get_vertex_labels():
                    options.append({
                        'query_part': f".hasLabel('{label}')",
                        'desc_part': f"，过滤出'{self.gremlin_base.get_schema_desc(label)}'类型的顶点",
                        'new_label': label, 'new_type': current_type
                    })

        elif step_name == 'drop':
            options.append({'query_part': ".drop()", 'desc_part': "，并删除它", 'new_label': None, 'new_type': 'none'})

        return options

    def _apply_random_enhancements(self, query: str, desc: str, current_label: str, current_type: str) -> List[Tuple[str, str]]:
        """
        对查询进行随机增强，添加一些通用的筛选和限制条件。
        
        Args:
            query: 当前查询字符串
            desc: 当前描述字符串  
            current_label: 当前标签
            current_type: 当前类型
            
        Returns:
            增强后的查询-描述对列表
        """
        enhanced_queries = []
        
        # 判断当前查询的状态，决定可以添加哪些增强
        if self._is_terminal_step(query):
            # 如果是终止步骤，不进行增强
            return enhanced_queries
            
        if self._is_element_stream(query, current_type):
            # 元素流状态：可以添加数量限制、去重、排序等
            enhanced_queries.extend(self._add_element_stream_enhancements(query, desc, current_label, current_type))
            
        elif self._is_value_stream(query):
            # 值流状态：可以添加去重、排序、数量限制
            enhanced_queries.extend(self._add_value_stream_enhancements(query, desc))
            
        return enhanced_queries
    
    def _is_terminal_step(self, query: str) -> bool:
        """判断查询是否以终止步骤结尾"""
        terminal_steps = ['.count()', '.sum()', '.mean()', '.min()', '.max()', '.drop()', '.iterate()']
        return any(query.endswith(step) for step in terminal_steps)
    
    def _is_element_stream(self, query: str, current_type: str) -> bool:
        """判断当前是否为元素流（顶点或边的流）"""
        # 如果当前类型是vertex或edge，且不是值流，则为元素流
        return current_type in ['vertex', 'edge'] and not self._is_value_stream(query)
    
    def _is_value_stream(self, query: str) -> bool:
        """判断当前是否为值流"""
        value_steps = ['.values(', '.valueMap(', '.id()', '.label()', '.key()']
        return any(step in query for step in value_steps)
    
    def _add_element_stream_enhancements(self, query: str, desc: str, current_label: str, current_type: str) -> List[Tuple[str, str]]:
        """为元素流添加增强"""
        enhancements = []
        
        # 1. 数量限制 - limit(n)
        if random.random() < 0.4:  # 40% 概率添加limit
            # 使用更广泛但仍然合理的范围
            if random.random() < 0.7:  # 70%概率使用常见值
                limit_num = random.choice([1, 3, 5, 10, 20, 50, 100])
            else:  # 30%概率使用随机值
                limit_num = random.randint(1, 200)  # 限制在合理范围内
            enhanced_query = f"{query}.limit({limit_num})"
            enhanced_desc = f"{desc}，并只取前{limit_num}个结果"
            enhancements.append((enhanced_query, enhanced_desc))
        
        # 2. 范围限制 - range(low, high)  
        if random.random() < 0.2:  # 20% 概率添加range
            if random.random() < 0.6:  # 60%概率使用常见值
                low = random.choice([0, 1, 5, 10, 20])
                high = low + random.choice([5, 10, 15, 20, 30])
            else:  # 40%概率使用随机值
                low = random.randint(0, 50)
                high = low + random.randint(5, 100)
            enhanced_query = f"{query}.range({low}, {high})"
            enhanced_desc = f"{desc}，并获取第{low}到{high}个结果"
            enhancements.append((enhanced_query, enhanced_desc))
        
        # 3. 随机采样 - sample(n)
        if random.random() < 0.3:  # 30% 概率添加sample
            if random.random() < 0.8:  # 80%概率使用常见值
                sample_num = random.choice([1, 2, 3, 5, 10])
            else:  # 20%概率使用随机值
                sample_num = random.randint(1, 50)
            enhanced_query = f"{query}.sample({sample_num})"
            enhanced_desc = f"{desc}，并随机抽取{sample_num}个"
            enhancements.append((enhanced_query, enhanced_desc))
        
        # 4. 去重 - dedup()
        if random.random() < 0.3:  # 30% 概率添加dedup
            enhanced_query = f"{query}.dedup()"
            enhanced_desc = f"{desc}，并去除重复项"
            enhancements.append((enhanced_query, enhanced_desc))
        
        # 5. 简单排序 - order() (不使用by子句，避免复杂性)
        if current_label and random.random() < 0.2:  # 20% 概率添加简单排序
            # 只有在有明确标签时才尝试排序
            sortable_props = self._get_sortable_properties(current_label)
            if sortable_props:
                prop = random.choice(sortable_props)
                order_dir = random.choice(['asc', 'desc'])
                enhanced_query = f"{query}.order().by('{prop}', {order_dir})"
                enhanced_desc = f"{desc}，并按{self.gremlin_base.get_schema_desc(prop)}{'升序' if order_dir == 'asc' else '降序'}排列"
                enhancements.append((enhanced_query, enhanced_desc))
        
        return enhancements
    
    def _add_value_stream_enhancements(self, query: str, desc: str) -> List[Tuple[str, str]]:
        """为值流添加增强"""
        enhancements = []
        
        # 1. 数量限制
        if random.random() < 0.4:
            if random.random() < 0.7:  # 70%概率使用常见值
                limit_num = random.choice([1, 3, 5, 10, 20])
            else:  # 30%概率使用随机值
                limit_num = random.randint(1, 100)
            enhanced_query = f"{query}.limit({limit_num})"
            enhanced_desc = f"{desc}，并只取前{limit_num}个值"
            enhancements.append((enhanced_query, enhanced_desc))
        
        # 2. 去重
        if random.random() < 0.4:
            enhanced_query = f"{query}.dedup()"
            enhanced_desc = f"{desc}，并去除重复的值"
            enhancements.append((enhanced_query, enhanced_desc))
        
        # 3. 排序（值流可以直接排序，不需要by子句）
        if random.random() < 0.3:
            enhanced_query = f"{query}.order()"
            enhanced_desc = f"{desc}，并按字母/数字顺序排列"
            enhancements.append((enhanced_query, enhanced_desc))
        
        return enhancements
    
    def _get_sortable_properties(self, label: str) -> List[str]:
        """获取指定标签的可排序属性"""
        if not label:
            return []
            
        try:
            properties = self.schema.get_properties_with_type(label)
            # 选择数值型和字符串型属性作为可排序属性
            sortable = []
            for prop in properties:
                if prop['type'] in ['STRING', 'INT32', 'INT64', 'FLOAT', 'DOUBLE']:
                    sortable.append(prop['name'])
            return sortable
        except:
            return []

# --- 单模块测试入口 ---
if __name__ == "__main__":
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(base_dir)
    # schema_path = os.path.join(project_root, 'db_data', 'schema', 'movie_schema.json')
    # data_path = os.path.join(project_root, 'db_data', 'movie', 'raw_data')
    # config_path = os.path.join(project_root, 'config.json')
    project_root = "/root/lzj/ospp/AST_Text2Gremlin"
    schema_path = os.path.join(project_root, 'db_data', 'schema', 'movie_schema.json')
    data_path = os.path.join(project_root, 'db_data')
    config_path = "/root/lzj/ospp/AST_Text2Gremlin/config.json"

    if not all(os.path.exists(p) for p in [schema_path, data_path, config_path]):
        print("错误: 找不到关键文件，请检查路径。")
    else:
        config = Config(file_path=config_path)
        schema = Schema(schema_path, data_path)
        gremlin_base = GremlinBase(config)
        
        print("\n--- [最终递归版] 测试'查询'配方: g.V().has('name', '...').out('acted_in') ---")
        query_recipe = Traversal()
        query_recipe.add_step(Step('V'))
        query_recipe.add_step(Step('has', params=['name', 'some_value']))
        query_recipe.add_step(Step('out', params=['acted_in']))
        
        generator = TraversalGenerator(schema, query_recipe, gremlin_base)
        generated_queries = generator.generate()
        print(f"查询配方共生成了 {len(generated_queries)} 条不同的语料。部分示例如下:")
        for i, (q, d) in enumerate(random.sample(generated_queries, min(5, len(generated_queries)))):
            print(f"  实例 {i+1}:\n    查询: {q}\n    描述: {d}")

        print("\n--- [最终递归版] 测试'增加'配-方: g.addV('person') ---")
        add_recipe = Traversal()
        add_recipe.add_step(Step('addV', params=['person']))
        
        generator = TraversalGenerator(schema, add_recipe, gremlin_base)
        generated_adds = generator.generate()
        print(f"增加配方共生成了 {len(generated_adds)} 条不同的语料。部分示例如下:")
        for i, (q, d) in enumerate(random.sample(generated_adds, min(5, len(generated_adds)))):
            print(f"  实例 {i+1}:\n    查询: {q}\n    描述: {d}")
        
        print("\n--- [最终递归版] 测试'更新'配方: g.V().property('born', ...) ---")
        update_recipe = Traversal()
        update_recipe.add_step(Step('V'))
        update_recipe.add_step(Step('property', params=['born', 1960]))

        generator = TraversalGenerator(schema, update_recipe, gremlin_base)
        generated_updates = generator.generate()
        print(f"更新配方共生成了 {len(generated_updates)} 条不同的语料。部分示例如下:")
        for i, (q, d) in enumerate(random.sample(generated_updates, min(5, len(generated_updates)))):
            print(f"  实例 {i+1}:\n    查询: {q}\n    描述: {d}")

        print("\n--- [最终递归版] 测试'删除'配方: g.V().has('name', '...').drop() ---")
        drop_recipe = Traversal()
        drop_recipe.add_step(Step('V'))
        drop_recipe.add_step(Step('has', params=['name', 'some_value']))
        drop_recipe.add_step(Step('drop'))
        
        generator = TraversalGenerator(schema, drop_recipe, gremlin_base)
        generated_drops = generator.generate()
        print(f"删除配方共生成了 {len(generated_drops)} 条不同的语料。部分示例如下:")
        for i, (q, d) in enumerate(random.sample(generated_drops, min(5, len(generated_drops)))):
            print(f"  实例 {i+1}:\n    查询: {q}\n    描述: {d}")
