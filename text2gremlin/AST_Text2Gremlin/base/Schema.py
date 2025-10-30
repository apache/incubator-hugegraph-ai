
"""
图数据库Schema管理模块。

负责解析Schema定义和CSV数据文件，为查询生成器提供图结构信息和真实数据实例。
"""

import os
import json
import pandas as pd
from typing import List, Dict
import json
import random
import pandas as pd
from typing import List, Dict, Any, Tuple

class Schema:
    def __init__(self, schema_file: str, data_dir: str):
        self.data_dir = data_dir
        self.vertices: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}
        self.vertex_data: Dict[str, pd.DataFrame] = {}
        self.edge_data: Dict[str, pd.DataFrame] = {}

        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)

        # 解析 schema 定义
        for item in schema_data.get('schema', []):
            label = item['label']
            if item['type'] == 'VERTEX':
                self.vertices[label] = {
                    'primary': item.get('primary', None),
                    'properties': {prop['name']: {'type': prop['type'], 'optional': prop.get('optional', False)} for prop in item.get('properties', [])}
                }
            elif item['type'] == 'EDGE':
                self.edges[label] = {
                    'source': None, 'destination': None,
                    'properties': {prop['name']: {'type': prop['type'], 'optional': prop.get('optional', False)} for prop in item.get('properties', [])}
                }
        
        # 2. 解析 files 定义，获取路径、header行数和边的端点
        self.vertex_files: Dict[str, Dict] = {}
        self.edge_files: Dict[str, Dict] = {}
        for file_info in schema_data.get('files', []):
            label = file_info['label']
            path = os.path.join(self.data_dir, file_info['path'])
            header_rows = file_info.get('header', 1) # 获取header行数，默认为1

            file_details = {'path': path, 'header_rows': header_rows}

            is_edge = 'SRC_ID' in file_info and 'DST_ID' in file_info
            if is_edge:
                self.edge_files[label] = file_details
                if label in self.edges:
                    self.edges[label]['source'] = file_info['SRC_ID']
                    self.edges[label]['destination'] = file_info['DST_ID']
            else:
                self.vertex_files[label] = file_details

    def _parse_custom_csv(self, file_path: str, header_line_index: int) -> pd.DataFrame:
        """解析自定义多行表头的 CSV 文件。"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 从第二行解析列名
            header_line = lines[header_line_index - 1]
            column_defs = header_line.strip().split(',')
            column_names = [d.split(':')[0] for d in column_defs]
            
            # 处理重复的列名（为重复的列添加后缀）
            seen = {}
            unique_names = []
            for name in column_names:
                if name in seen:
                    seen[name] += 1
                    unique_names.append(f"{name}_{seen[name]}")
                else:
                    seen[name] = 0
                    unique_names.append(name)
            column_names = unique_names

            # 从指定header行之后开始读取数据
            data_lines = lines[header_line_index:]
            
            if not data_lines:
                return pd.DataFrame(columns=column_names)

            # 使用pandas从内存中的字符串列表读取数据
            from io import StringIO
            csv_data = StringIO("".join(data_lines))
            df = pd.read_csv(csv_data, header=None, names=column_names)
            return df

        except (FileNotFoundError, IndexError) as e:
            print(f"警告: 读取或解析文件失败: {file_path}, 错误: {e}")
            return pd.DataFrame()

    def _load_vertex_data(self, label: str):
        if label not in self.vertex_data and label in self.vertex_files:
            file_info = self.vertex_files[label]
            self.vertex_data[label] = self._parse_custom_csv(file_info['path'], file_info['header_rows'])

    def _load_edge_data(self, label: str):
        if label not in self.edge_data and label in self.edge_files:
            file_info = self.edge_files[label]
            self.edge_data[label] = self._parse_custom_csv(file_info['path'], file_info['header_rows'])

    # --- Schema 查询方法 (保持不变) ---
    def get_vertex_labels(self) -> List[str]:
        return list(self.vertices.keys())

    def get_edge_labels(self) -> List[str]:
        return list(self.edges.keys())

    def get_properties_with_type(self, label: str) -> List[Dict[str, str]]:
        props_dict = self.vertices.get(label, {}).get('properties', {}) or self.edges.get(label, {}).get('properties', {})
        return [{'name': name, 'type': meta['type']} for name, meta in props_dict.items()]

    def get_valid_steps(self, current_label: str, element_type: str = 'vertex') -> List[Dict]:
        if element_type == 'vertex':
            if current_label not in self.vertices: return []
            valid_steps = []
            outgoing = [l for l, e in self.edges.items() if e['source'] == current_label]
            if outgoing: valid_steps.append({'step': 'out', 'params': outgoing})
            incoming = [l for l, e in self.edges.items() if e['destination'] == current_label]
            if incoming: valid_steps.append({'step': 'in', 'params': incoming})
            props = self.get_properties_with_type(current_label)
            if props:
                valid_steps.append({'step': 'properties', 'params': [p['name'] for p in props]})
                valid_steps.append({'step': 'has', 'params': props})
            return valid_steps
        return []

    def get_step_result_label(self, start_label: str, step: Dict) -> Tuple[str, str]:
        step_name, step_param = step.get('step'), step.get('param')
        if step_name == 'out': return self.edges[step_param]['destination'], 'vertex'
        if step_name == 'in': return self.edges[step_param]['source'], 'vertex'
        if step_name in ['properties', 'has', 'values']: return start_label, 'vertex'
        return None, None

    def get_vertex_creation_info(self, label: str) -> Dict:
        if label not in self.vertices: return {}
        schema_info = self.vertices[label]
        required = [name for name, meta in schema_info['properties'].items() if not meta['optional']]
        return {'primary': schema_info.get('primary'), 'required': required}

    def get_edge_creation_info(self, label: str) -> Tuple[str, str]:
        if label in self.edges: return (self.edges[label]['source'], self.edges[label]['destination'])
        return (None, None)

    def get_updatable_properties(self, label: str) -> List[Dict[str, str]]:
        if label not in self.vertices: return []
        schema_info = self.vertices[label]
        primary_key = schema_info.get('primary')
        return [{'name': name, 'type': meta['type']} for name, meta in schema_info['properties'].items() if name != primary_key]

    def get_instance(self, label: str) -> Dict:
        """获取单个实例（保持向后兼容）"""
        instances = self.get_instances(label, count=1)
        return instances[0] if instances else {}
    
    def get_instances(self, label: str, count: int = None) -> List[Dict]:
        """获取多个实例
        
        Args:
            label: 标签名
            count: 要获取的实例数量，如果为None则随机选择2-5个
            
        Returns:
            实例列表
        """
        import random
        
        is_edge = label in self.edges
        data_cache = self.edge_data if is_edge else self.vertex_data
        load_func = self._load_edge_data if is_edge else self._load_vertex_data
        
        if label not in data_cache: 
            load_func(label)
        
        df = data_cache.get(label)
        if df is None or df.empty:
            return []
        
        # 如果没有指定数量，随机选择2-5个
        if count is None:
            count = random.randint(2, 5)
        
        # 如果实际数据量小于要求的数量，就全部取出
        actual_count = min(count, len(df))
        
        # 随机采样
        sampled_df = df.sample(actual_count)
        return sampled_df.to_dict('records')

# --- 单模块测试入口 ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    schema_path = os.path.join(project_root, 'db_data', 'schema', 'movie_schema.json')
    # 【修正】将 data_path 指向包含 movie/raw_data 的上级目录
    data_path = os.path.join(project_root, 'db_data')

    if not os.path.exists(schema_path) or not os.path.exists(data_path):
        print("错误: 找不到 Schema 或数据文件，请检查路径。")
    else:
        schema = Schema(schema_path, data_path)
        print("\n--- Schema 初始化成功 (已修复CSV读取逻辑) ---")

        print("\n--- 测试数据实例获取 ---")
        random_person = schema.get_instance('person')
        print(f"随机获取一个 'person' 实例: {random_person}")
        
        random_user = schema.get_instance('user')
        print(f"随机获取一个 'user' 实例: {random_user}")
        
        # 验证 name 属性是否能被正确读取
        if random_person and 'name' in random_person:
            print(f"成功读取到 'person' 的 name: {random_person['name']}")
        else:
            print("错误: 未能从 'person' 实例中读取到 name 属性。")