## 处理处理书籍
import re
import json
from typing import List, Dict, Any

def parse_markdown_with_hierarchical_levels(
    file_path: str, 
    output_path: str = "gremlin_book_hierarchical.json"
) -> None:
    """
    读取Markdown文件，按三级标题结构进行解析，并为每个层级生成
    小数形式的编号（如1, 1.1, 1.1.1），最终输出为层级化的JSON文件。

    Args:
        file_path (str): 输入的Markdown文件路径 )。
        output_path (str): 输出的JSON文件名。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误：文件未找到于 {file_path}")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    book_structure: List[Dict[str, Any]] = []
    
    # 初始化层级计数器
    level1_counter = 0
    level2_counter = 0
    level3_counter = 0
    
    # 追踪当前所在的节点
    current_chapter: Dict[str, Any] = {}
    current_section_lvl2: Dict[str, Any] = {}
    current_section_lvl3: Dict[str, Any] = {}

    def add_content_to_current_node(line: str):
        if current_section_lvl3:
            current_section_lvl3["content"].append(line)
        elif current_section_lvl2:
            current_section_lvl2["content"].append(line)
        elif current_chapter:
            current_chapter["content"].append(line)
    #可能出现误差，比如存在"#"注释，最好人工筛查一遍
    for line in lines:
        if line.startswith('# '):
            level1_counter += 1
            level2_counter = 0
            level3_counter = 0
            
            level_str = f"{level1_counter}"
            title = line.strip('# \n')
            
            current_section_lvl2 = {}
            current_section_lvl3 = {}
            current_chapter = {
                "level": level_str,
                "title": f"{level_str}. {title}",
                "content": [],
                "children": []
            }
            book_structure.append(current_chapter)
            
        elif line.startswith('## '):
            level2_counter += 1
            level3_counter = 0
            
            level_str = f"{level1_counter}.{level2_counter}"
            title = line.strip('# \n')

            current_section_lvl3 = {}
            current_section_lvl2 = {
                "level": level_str,
                "title": f"{level_str}. {title}",
                "content": [],
                "children": []
            }
            if current_chapter:
                current_chapter["children"].append(current_section_lvl2)

        elif line.startswith('### '):
            level3_counter += 1
            
            level_str = f"{level1_counter}.{level2_counter}.{level3_counter}"
            title = line.strip('# \n')
            
            current_section_lvl3 = {
                "level": level_str,
                "title": f"{level_str}. {title}",
                "content": [],
                "children": []
            }
            if current_section_lvl2:
                current_section_lvl2["children"].append(current_section_lvl3)

        else:
            add_content_to_current_node(line)

    def finalize_content(node: Dict[str, Any]):
        node['content'] = ''.join(node['content']).strip()
        for child in node.get('children', []):
            finalize_content(child)

    for chapter in book_structure:
        finalize_content(chapter)
        
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(book_structure, f, ensure_ascii=False, indent=2)
        print(f"✅ 书籍内容已成功提取并保存到: {output_path}")
    except Exception as e:
        print(f"写入JSON文件时发生错误: {e}")

# def parse_markdown_with_hierarchical_levels(
#     file_path: str, 
#     output_path: str = "gremlin_book_hierarchical.json"
# ) -> None:
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
#     except FileNotFoundError:
#         print(f"错误：文件未找到于 {file_path}")
#         return
#     except Exception as e:
#         print(f"读取文件时发生错误: {e}")
#         return

#     book_structure: List[Dict[str, Any]] = []
    
#     # 初始化层级计数器
#     level_counters = [0, 0, 0] 
#     current_nodes: List[Dict[str, Any]] = [None, None, None]

#     # 判断是否在代码块内
#     in_code_block = False

#     def add_content_to_current_node(line: str):
#         for i in range(2, -1, -1):
#             if current_nodes[i]:
#                 current_nodes[i]["content"].append(line)
#                 return

#     for line in lines:
#         if line.strip().startswith('```'):
#             in_code_block = not in_code_block
#             add_content_to_current_node(line)
#             continue 

#         # 只有不在代码块内时，才进行标题匹配
#         matched_heading = False
#         if not in_code_block:
#             # 匹配1到3级标题
#             match = re.match(r'^(#{1,3})\s(.*)', line)
#             if match:
#                 level = len(match.group(1)) - 1 
#                 title = match.group(2).strip()
                
#                 # 更新层级计数器
#                 level_counters[level] += 1
#                 for i in range(level + 1, 3):
#                     level_counters[i] = 0 # 重置更深层级的计数器

                
#                 level_str = ".".join(map(str, [c for c in level_counters[:level+1] if c > 0]))

#                 new_node = {
#                     "level": level_str,
#                     "title": f"{level_str}. {title}",
#                     "content": [],
#                     "children": []
#                 }
                
#                 # 将新节点挂载到正确的父节点下
#                 if level == 0: 
#                     book_structure.append(new_node)
#                 elif level > 0 and current_nodes[level - 1]: 
#                     current_nodes[level - 1]["children"].append(new_node)
                
#                 # 更新当前节点指针
#                 current_nodes[level] = new_node
#                 matched_heading = True

#         # 如果当前行不是标题，或者在代码块内，则视为内容
#         if not matched_heading:
#             add_content_to_current_node(line)

#     # 递归，用于将收集到的内容行列表合并成单个字符串
#     def finalize_content(node: Dict[str, Any]):
#         node['content'] = ''.join(node['content']).strip()
#         for child in node.get('children', []):
#             finalize_content(child)

#     for chapter in book_structure:
#         finalize_content(chapter)
        
#     try:
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(book_structure, f, ensure_ascii=False, indent=2)
#         print(f"✅ 书籍内容已成功提取并保存到: {output_path}")
#     except Exception as e:
#         print(f"写入JSON文件时发生错误: {e}")

if __name__ == "__main__":
    md_file_name = "./data/book/Gremlin-Graph-Guide.md"
    output_json_file = "./data/book/gremlin_book.json"
    parse_markdown_with_hierarchical_levels(md_file_name, output_json_file)