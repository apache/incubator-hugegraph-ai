
"""
Gremlin翻译引擎模块。

提供Gremlin术语到中文的智能翻译，负责生成自然流畅的中文查询描述。
"""

import os
import random
from gremlin.GremlinParser import GremlinParser

class GremlinBase:
    def __init__(self, config):
        """
        Gremlin 基础类，作为项目的“工具箱”和“字典”。
        """
        self.config = config
        
        # 从 GremlinParser 加载rule_names
        self.rule_names = GremlinParser.ruleNames

        self.token_dict = {}
        self.template = [] # 索引对应 token_dict 的值，每个元素是一个包含多种中文翻译模板的子列表。
        self._initialize_translation_templates()

        # 复用 schema_dict 加载同义词等
        self.schema_dict = {}
        self._load_schema_translations()

    def get_rule_name(self, rule_index: int) -> str:
        """根据索引获取 ANTLR 规则名。"""
        if 0 <= rule_index < len(self.rule_names):
            return self.rule_names[rule_index]
        return "UnknownRule"

    def _load_schema_translations(self):
        """加载schema翻译字典"""
        # 从Config获取路径，如果失败则使用默认路径
        file_paths = []
        
        try:
            if hasattr(self.config, 'get_schema_dict_path'):
                schema_dict_paths = self.config.get_schema_dict_path()
                # 为列表或字符串的情况
                if isinstance(schema_dict_paths, list):
                    file_paths.extend(schema_dict_paths)
                elif isinstance(schema_dict_paths, str):
                    file_paths.append(schema_dict_paths)
            
            if hasattr(self.config, 'get_syn_dict_path'):
                syn_dict_path = self.config.get_syn_dict_path()
                if syn_dict_path:
                    file_paths.append(syn_dict_path)
                    
        except Exception as e:
            print(f"[INFO] Config paths not available: {e}")
        
        # 如果没有从Config获取到路径，使用默认路径
        if not file_paths:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_paths = [
                os.path.join(current_dir, 'template', 'schema_dict.txt'),
                os.path.join(current_dir, 'template', 'syn_dict.txt')
            ]
        
        # 加载schema翻译字典
        existing_paths = [path for path in file_paths if os.path.exists(path)]
        
        # 如果没有找到配置的路径，尝试默认路径
        if not existing_paths:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            default_paths = [
                os.path.join(current_dir, 'template', 'schema_dict.txt'),
                os.path.join(current_dir, 'template', 'syn_dict.txt')
            ]
            existing_paths = [path for path in default_paths if os.path.exists(path)]
            if existing_paths:
                print(f"[INFO] Using default dictionary paths: {existing_paths}")
        
        if existing_paths:
            self.load_dict_from_file(existing_paths)
        else:
            print(f"[WARNING] No dictionary files found in: {file_paths}")

    def _initialize_translation_templates(self):
        """
        初始化 Gremlin 步骤的翻译模板。
        """
        # 定义模板数据
        templates_data = {
            # --- 起始步骤 ---
            "v": ["查询图中的所有顶点", "获取所有节点"],
            "e": ["查询图中的所有边", "获取所有关系"],
            "addv": ["添加一个标签为 '{}' 的新顶点"],
            "adde": ["添加一条从一个顶点到另一个顶点的 '{}' 边"],
            # --- 遍历步骤 ---
            "out": ["从当前位置出发，沿着 '{}' 方向的出边前进", "找到 '{}' 类型的邻居"],
            "in": ["从当前位置出发，沿着 '{}' 方向的入边前进", "找到拥有 '{}' 类型关系的来源"],
            "both": ["沿着 '{}' 方向的双向边进行遍历"],
            "outv": ["从当前边，找到它的出射顶点", "获取边的头节点"],
            "inv": ["从当前边，找到它的入射顶点", "获取边的尾节点"],
            "bothv": ["从当前边，找到它的两个端点"],
            # --- 过滤步骤 ---
            "haslabel": ["并筛选出标签为 '{}' 的元素"],
            "has": ["并筛选出属性 '{}' 为 '{}' 的元素", "查找其中 '{}' 是 '{}' 的数据"],
            "where": ["并根据 '{}' 的条件进行过滤"],
            "limit": ["并限制最多返回 {} 个结果", "取前 {} 条数据"],
            "dedup": ["并对结果进行去重"],
            # --- 映射/返回步骤 ---
            "values": ["然后获取它们的 '{}' 属性值", "提取 '{}' 字段的值"],
            "valuemap": ["然后以键值对的形式返回它们的属性"],
            "label": ["然后获取它们的标签"],
            "id": ["然后获取它们的ID"],
            "path": ["然后返回完整的遍历路径"],
            "project": ["然后将结果投影为以 '{}' 为键的映射"],
            "by": ["通过 '{}' 来进行分组或投影"],
            # --- 聚合/排序步骤 ---
            "count": ["最后统计结果的总数"],
            "group": ["然后根据 '{}' 进行分组"],
            "order": ["然后对结果进行排序"],
            # --- 修改/删除步骤 ---
            "property": ["并将其 '{}' 属性的值更新为 '{}'"],
            "drop": ["最后将这些元素从图中删除", "移除这些数据"]
        }
        
        # 填充 self.token_dict 和 self.template
        for index, (key, value) in enumerate(templates_data.items()):
            self.token_dict[key.upper()] = index
            self.template.append(value)

    def get_token_desc(self, token_key: str, *args) -> str:
        """
        根据 token 和参数获取一个随机的、格式化后的中文描述。
        """
        key = token_key.upper()
        if key in self.token_dict:
            index = self.token_dict[key]
            # 随机选择一个模板
            selected_template = random.choice(self.template[index])
            try:
                # 翻译参数中的schema术语
                translated_args = []
                for arg in args:
                    if isinstance(arg, str):
                        # 尝试翻译schema术语
                        translated_arg = self.get_schema_desc(arg)
                        translated_args.append(translated_arg)
                    else:
                        translated_args.append(arg)
                
                # 使用翻译后的参数格式化模板
                return selected_template.format(*translated_args)
            except (IndexError, KeyError):
                # 如果参数数量不匹配，返回原始模板
                return selected_template
        return "" # 如果 token 不存在，返回空字符串

    # 复用的通用方法
    def merge_desc(self, desc_list: list) -> str:
        """合并多个描述片段，移除空字符串并用合适的连接词连接。"""
        # 过滤掉空字符串
        filtered_list = [s for s in desc_list if s and s.strip()]
        return ",".join(filtered_list)

    def load_dict_from_file(self, file_paths: list):
        """从文件加载字典，例如同义词词典。"""
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"[WARNING] Dictionary file not found: {file_path}")
                continue
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    elements = line.strip().split()
                    if elements:
                        key = elements[0]
                        values = elements[1:]
                        self.schema_dict[key] = values

    def get_schema_desc(self, key: str) -> str:
        """从加载的字典中获取一个随机的同义词或描述。"""
        try:
            # 确保键存在
            if key in self.schema_dict and self.schema_dict[key]:
                return random.choice(self.schema_dict[key])
            return key # 如果没有同义词，返回原词
        except KeyError:
            return key

if __name__ == "__main__":
    # 临时创建config 对象，用于测试
    class MockConfig:
        def get_schema_dict_path(self):
            return "./template/schema_dict.txt" 
        def get_syn_dict_path(self):
            return "./template/syn_dict.txt" 

    config = MockConfig()
    gremlin_base = GremlinBase(config)

    print("--- GremlinBase.py 测试 ---")
    
    # 1. 测试规则名加载
    print(f"\n成功加载 {len(gremlin_base.rule_names)} 条 Gremlin 语法规则。")
    print(f"第一条规则是: '{gremlin_base.get_rule_name(0)}'")

    # 2. 测试翻译模板
    print("\n--- 测试翻译功能 ---")
    print("OUT('acted_in') 的一个翻译: ", gremlin_base.get_token_desc("OUT", "acted_in"))
    print("HAS('name', 'marko') 的一个翻译: ", gremlin_base.get_token_desc("HAS", "name", "marko"))
    print("LIMIT(10) 的一个翻译: ", gremlin_base.get_token_desc("LIMIT", 10))
    print("COUNT() 的一个翻译: ", gremlin_base.get_token_desc("COUNT"))
    
    # 3. 测试描述合并
    desc_parts = ["查找所有电影", "筛选出其中类型为科幻的", "最后统计总数"]
    merged = gremlin_base.merge_desc(desc_parts)
    print("\n合并后的描述: ", merged)

