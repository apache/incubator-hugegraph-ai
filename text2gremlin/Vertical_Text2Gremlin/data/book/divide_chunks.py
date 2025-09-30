import json
import re
from typing import List, Dict, Any

# --- 核心参数 (您可以根据需要调整) ---
MIN_UNITS_THRESHOLD = 3
CHUNK_SIZE = 4
OVERLAP_SIZE = 1

def _get_semantic_units(text: str) -> List[str]:
    """
    将文本内容分解为“语义单元”列表。
    语义单元要么是一个“原子块”（代码块、列表等），要么是一个普通段落。
    """
    # 1. 定义所有原子块的正则表达式
    
    # --- BUG修复：将代码块的捕获组 (...) 修改为非捕获组 (?:...) ---
    code_block_pattern = r'(?:^```.*?^```)' 
    
    list_pattern = r'(?:(?:\n\n|^)(?:[ \t]*(?:\*|\-|\d+\.)[ \t].*(?:\n|$))+)'
    table_pattern = r'(?:(?:\n\n|^)(?:\|.*\|(?:\n|$))+)'
    blockquote_pattern = r'(?:(?:\n\n|^)(?:>.*(?:\n|$))+)'

    atomic_patterns = [code_block_pattern, list_pattern, table_pattern, blockquote_pattern]
    
    master_pattern = re.compile(f"({'|'.join(atomic_patterns)})", re.MULTILINE | re.DOTALL)
    
    parts = master_pattern.split(text)
    
    semantic_units = []
    for part in parts:
        # --- BUG修复：在执行 .strip() 前，先判断 part 是否为 None ---
        if part:
            stripped_part = part.strip()
            if stripped_part:
                semantic_units.append(stripped_part)
            
    return semantic_units

def _chunk_content_string(
    content: str, 
    hierarchy: Dict[str, Any], 
    chunk_id_prefix: str
) -> List[Dict[str, Any]]:
    if not content:
        return []

    semantic_units = _get_semantic_units(content)
    
    # 粗略计算单词数以应用阈值
    word_count_approx = len(re.findall(r'\w+', "".join(semantic_units)))

    if len(semantic_units) < MIN_UNITS_THRESHOLD or word_count_approx < 100:
        chunk_content = "\n\n".join(semantic_units)
        metadata = {
            "source_file": "Gremlin-Graph-Guide.md",
            "hierarchy": hierarchy,
            "chunk_id": f"{chunk_id_prefix}_chunk_1",
            "word_count": len(re.findall(r'\w+', chunk_content))
        }
        return [{"chunk_content": chunk_content, "metadata": metadata}]

    chunks = []
    stride = CHUNK_SIZE - OVERLAP_SIZE
    if stride <= 0:
        stride = 1

    for i in range(0, len(semantic_units), stride):
        window = semantic_units[i : i + CHUNK_SIZE]
        if not window:
            continue

        chunk_content = "\n\n".join(window)
        chunk_id = f"{chunk_id_prefix}_chunk_{i//stride + 1}"
        
        metadata = {
            "source_file": "Gremlin-Graph-Guide.md",
            "hierarchy": hierarchy,
            "chunk_id": chunk_id,
            "word_count": len(re.findall(r'\w+', chunk_content))
        }
        chunks.append({"chunk_content": chunk_content, "metadata": metadata})
        
        if i + CHUNK_SIZE >= len(semantic_units):
            break
            
    return chunks

def _traverse_and_chunk_node(
    node: Dict[str, Any], 
    parent_hierarchy: Dict[str, Any], 
    final_chunks: List[Dict[str, Any]]
):
    current_hierarchy = parent_hierarchy.copy()
    level_str = node.get("level", "")
    if level_str:
        level_num = len(level_str.split('.'))
        current_hierarchy[f"level_{level_num}_title"] = node.get("title", "")
    
    chunk_id_prefix = f"C{level_str.replace('.', '_')}"

    if node.get("content"):
        chunks_from_content = _chunk_content_string(node["content"], current_hierarchy, chunk_id_prefix)
        final_chunks.extend(chunks_from_content)
        
    for child_node in node.get("children", []):
        _traverse_and_chunk_node(child_node, current_hierarchy, final_chunks)

def create_fine_grained_chunks(
    input_json_path: str, 
    output_jsonl_path: str
):
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            book_structure = json.load(f)
    except Exception as e:
        print(f"错误：读取输入文件 '{input_json_path}' 失败: {e}")
        return

    final_chunks = []
    for chapter_node in book_structure:
        _traverse_and_chunk_node(chapter_node, {}, final_chunks)
    # 生成jsonl文件
    try:
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for chunk in final_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        print(f"✅ 成功生成 {len(final_chunks)} 个细粒度数据块，已保存到: {output_jsonl_path}")
    except Exception as e:
        print(f"错误：写入输出文件 '{output_jsonl_path}' 失败: {e}")

    # # 生成json文件
    # try:
    #     # 这里是将整个列表一次性写入，并使用 indent 参数美化格式
    #     with open(output_jsonl_path, 'w', encoding='utf-8') as f:
    #         json.dump(final_chunks, f, ensure_ascii=False, indent=2)
    #     print(f"✅ 成功生成 {len(final_chunks)} 个细粒度数据块，已保存到: {output_jsonl_path}")
    # except Exception as e:
    #     print(f"错误：写入输出文件 '{output_jsonl_path}' 失败: {e}")

# --- 执行脚本 ---
input_file = "gremlin_book.json"
output_file = "gremlin_book_chunks.jsonl"
# output_file = "gremlin_book_chunks.json"
create_fine_grained_chunks(input_file, output_file)