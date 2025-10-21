
import pandas as pd
import time
from typing import List, Dict
from gremlin_checker import check_gremlin_syntax
from llm_handler import generate_gremlin_variations, generate_texts_for_gremlin

INPUT_CSV_PATH = 'test_gremlin_qa_dataset.csv'  # 种子qa数据
OUTPUT_CSV_PATH = 'augmented_text2gremlin.csv' # 输出路径
CHUNK_SIZE = 100        # 每次从CSV中读取的行数
WRITE_THRESHOLD = 200   # 缓冲区中累积多少条新数据，持久化写入一次
GROUP_SIZE = 5          # 泛化时参考的同组问题数量



def save_and_clear_buffer(buffer: List[Dict], is_first_write: bool) -> bool:
    """
    将缓冲区数据进行持久化，并清空缓冲区。
    """
    if not buffer:
        return is_first_write

    print(f"\n--- 缓冲区达到阈值，正在写入 {len(buffer)} 条数据 ---")
    df_new = pd.DataFrame(buffer)
    df_new.to_csv(
        OUTPUT_CSV_PATH, 
        mode='a', 
        header=is_first_write, 
        index=False,
        encoding='utf-8-sig'
    )
    print(f"✅ 成功保存到 {OUTPUT_CSV_PATH}")
    buffer.clear()
    
    return False

def process_group(gremlin_query: str, group_df: pd.DataFrame) -> List[Dict]:
    """
    处理单个完整的分组
    """
    print("\n" + "="*80)
    print(f"正在处理分组: {gremlin_query[:100]}...")

    # 提取种子数据
    seed_questions = group_df['question'].tolist()[:GROUP_SIZE]

    # 生成Gremlin
    print(f"   Step 1: 正在调用LLM基于 {len(seed_questions)} 个问题生成Gremlin变体")
    candidate_queries = generate_gremlin_variations(gremlin_query, seed_questions)
    if not candidate_queries:
        print("   -> LLM 未返回Gremlin变体，跳过此分组。")
        return []
    
    print(f"   -> LLM 生成了 {len(candidate_queries)} 条候选查询:")

    # AST语法检查
    valid_queries = []
    for query in candidate_queries:
        is_valid, msg = check_gremlin_syntax(query)
        if is_valid:
            valid_queries.append(query)
            print(f"      语法正确: {query}")
        else:
            print(f"      ❌ 语法失败: {query} | 原因: {msg}")
    
    if not valid_queries:
        print("   -> 语法检查后无有效Gremlin,跳过此分组。")
        return []
    # 日志记录
    new_data_for_group = []
    print(f"   Step 2: 正在为 {len(valid_queries)} 条有效Gremlin生成Text")
    for valid_query in valid_queries:
        generated_texts = generate_texts_for_gremlin(valid_query)
        if generated_texts:
            print(f"      -> 为查询 '{valid_query[:80]}...' 生成了 {len(generated_texts)} 个问题。")
            for text in generated_texts:
                new_data_for_group.append({'question': text, 'gremlin_query': valid_query})
        time.sleep(1)

    return new_data_for_group

def main():
    is_first_write = True
    write_buffer = []
    carry_over_df = pd.DataFrame()  # 第一步的种子数据不会严格的按照要求的数量进行泛化，设置一个暂存区处理边界问题

    try:
        csv_reader = pd.read_csv(INPUT_CSV_PATH, chunksize=CHUNK_SIZE, iterator=True)
        
        for i, chunk_df in enumerate(csv_reader):
            print("\n" + "#"*30 + f" 开始处理数据块 Chunk {i+1} " + "#"*30)
            
            current_data = pd.concat([carry_over_df, chunk_df], ignore_index=True)
            if current_data.empty:
                continue

            last_query_in_chunk = current_data.iloc[-1]['gremlin_query']  # 找到末尾那个可能不完整的gremlin
            carry_over_df = current_data[current_data['gremlin_query'] == last_query_in_chunk].copy() # 暂存，留到下一轮检查
            df_to_process = current_data.drop(carry_over_df.index) 

            if not df_to_process.empty:
                grouped = df_to_process.groupby('gremlin_query',sort=False)
                for gremlin_query, group_df in grouped:
                    new_data = process_group(gremlin_query, group_df)
                    if new_data:
                        write_buffer.extend(new_data)
                    
                    if len(write_buffer) >= WRITE_THRESHOLD:
                        is_first_write = save_and_clear_buffer(write_buffer, is_first_write)

        print("\n" + "#"*30 + " 开始处理最后剩余的数据 " + "#"*30)
        if not carry_over_df.empty:
            final_grouped = carry_over_df.groupby('gremlin_query',sort=False)
            for gremlin_query, group_df in final_grouped:
                new_data = process_group(gremlin_query, group_df)
                if new_data:
                    write_buffer.extend(new_data)

        print("\n--- 正在执行最后的写入操作... ---")
        save_and_clear_buffer(write_buffer, is_first_write)

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{INPUT_CSV_PATH}'")
    except Exception as e:
        print(f"发生未知错误: {e}")

    print("\nQA泛化完成!")

if __name__ == '__main__':
    main()