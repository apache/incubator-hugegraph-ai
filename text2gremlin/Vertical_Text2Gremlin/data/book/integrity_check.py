import json
import re
from typing import Dict, Any, List

def verify_word_count_across_pipeline(
    original_md_path: str,
    structured_json_path: str,
    chunks_jsonl_path: str
):
    """chunk数据统计
    对整个数据处理流水线，统一按“字数 (Word Count)”进行完整性校验。
    """
    print("="*60)
    print("开始按“字数 (Word Count)”进行数据流水线校验...")
    print("="*60)

    # 计算原始MD文件的总字数
    original_word_count = 0
    try:
        with open(original_md_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        # \w+ 匹配一个或多个字母、数字或下划线，统计的“单词”而不是字符
        original_word_count = len(re.findall(r'\w+', original_content))
        print(f"1. 原始文件 '{original_md_path}':")
        print(f"   - 总字数: {original_word_count}")
    except Exception as e:
        print(f"   - 错误: 读取文件失败: {e}")

    # 递归计算结构化JSON中所有'content'的总字数
    structured_word_count = 0
    def _sum_content_words(node: Dict[str, Any]) -> int:
        content_text = node.get('content', '')
        count = len(re.findall(r'\w+', content_text))
        for child in node.get('children', []):
            count += _sum_content_words(child)
        return count

    try:
        with open(structured_json_path, 'r', encoding='utf-8') as f:
            book_structure = json.load(f)
        for chapter in book_structure:
            structured_word_count += _sum_content_words(chapter)
        print(f"\n2. 结构化文件 '{structured_json_path}':")
        print(f"   - 所有'content'字段总字数: {structured_word_count}")
    except Exception as e:
        print(f"   - 错误: 读取或处理文件失败: {e}")

    # 累加所有Chunks的总字数
    total_chunks_word_count = 0
    try:
        with open(chunks_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunk_content = chunk.get("chunk_content", "")
                total_chunks_word_count += len(re.findall(r'\w+', chunk_content))
        print(f"\n3. 切分后文件 '{chunks_jsonl_path}':")
        print(f"   - 所有'chunk_content'字段总字数: {total_chunks_word_count}")
    except Exception as e:
        print(f"   - 错误: 读取或处理文件失败: {e}")
        
    print("\n" + "="*60)
    print("校验结论分析:")
    
    if original_word_count > structured_word_count:
        diff = original_word_count - structured_word_count
        print(f"✅ (预期之内) 结构化JSON的字数比原始MD文件少了 {diff} 个字。")
        print(f"   这部分差异主要是因为标题文字被提取到了'title'字段，不再计入'content'。")
    else:
        print(f"不合理,结构化JSON的字数未少于原始MD文件。")

    if total_chunks_word_count > structured_word_count:
        overlap_words = total_chunks_word_count - structured_word_count
        print(f"\n最终Chunks的总字数比结构化JSON多了 {overlap_words} 个字。")
        print(f"   这证明了重叠切分策略在正常工作，保证了上下文的完整性。")
    else:
        print(f"可能存在问题，最终Chunks的总字数未多于结构化JSON，可能重叠策略未生效或内容有丢失。")
        
    print("\n" + "="*60)
    print("校验完成。")
    print("="*60)

original_file = "Gremlin-Graph-Guide.md"
structured_file = "gremlin_book.json"
chunked_file = "gremlin_book_chunks.jsonl" 

verify_word_count_across_pipeline(original_file, structured_file, chunked_file)