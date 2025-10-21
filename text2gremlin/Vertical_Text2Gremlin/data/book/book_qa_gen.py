import json
import os
import asyncio
from openai import AsyncOpenAI  
import time
from tenacity import retry, stop_after_attempt, wait_exponential 
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("ds_api_key") 
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

INPUT_FILE = "gremlin_book_chunks.jsonl"
# INPUT_FILE = "test.jsonl"
OUTPUT_FILE = "gremlin_qa_dataset_final_epoch_3.jsonl"


READ_BATCH_SIZE = 6     # 每次读取10条chunks进行并发处理
WRITE_BATCH_SIZE = 100    # 持久化阈值
RETRY_ATTEMPTS = 3       # 重试次数
RETRY_WAIT_SECONDS = 2   # 重试的初始等待时间，之后指数退避



SYSTEM_PROMPT = """
你是一位顶级的Gremlin及图数据库专家，你的任务是根据用户提供的技术文档片段，生成高质量的问答（Q&A）对，用于AI模型训练。

请严格遵守以下规则：
1. **生成多个问答对**: 如果原文中没有任何Gremlin相关的信息，不要输出任何信息。如果有Gremlin相关的信息，请基于提供的文本中所有关于gremlin的知识点，一般生成5到15条有价值的、内容涵盖文本中所有gremlin知识点的问答对,允许超过15条，因为第一原则是必须涵盖完提供的内容里所有关于gremlin的知识点。如果你判定内容里面有更多Gremlin的知识点需要被列举出,则必须生成更多的数据，直到涵盖完所给内容中里面所有与gremlin相关的知识点。
2. **忠于原文**: 原文文本的优先级是最高的，在原文有出处的情况下，所有的问题和答案都必须严格来源于提供的文本内容，如果原文没有，对于你非常明确的知识点，允许基于原文的基础上适当补充，但对于有疑虑或者不明确的地方，禁止使用任何外部知识或进行自由发挥。
3. **指代**:问题和答案尽量将重心放在Gremlin本身以及相关的知识中，而不是这本书或者作者之类的概念，如果必须要涉及到"这本书"、“作者”吗，请以"Gremlin官方教程"、"Gremlin官方作者"等代替。
4. **专注Gremlin知识**: 问题应该涵盖提供内容中所有与Gremlin有关的内容，对于Gremlin的核心语法、概念、函数用法或最佳实践等要重点聚焦，最好有相关使用例子，你的例子可以从提供文本中来或者在不影响正确的情况下稍微改写,也可以基于所给的文本容进行合理推断，但必须基于原文推断。如果给不出示例的情况，不要强行给示例，也不要在输出中出现"文本中没有相关示例..."这类信息。
5. **问题完整性**: 你的输出内容的使用者是无法看到原文的，所以你的问题和答案的上下文要完整，禁止出现后面这类指代不明确表达："文本中"、"提供的文本中的..."、"查询示例在提供的文本中给出..."、"提供的文本中..."、"原文中..."。
6. **简洁准确**: 答案应该像技术文档一样，直接、准确地回答问题。准确是第一原则，如果问题比较复杂，允许更多的回答内容，不能为了简洁地省略某些内容，在保证准确的基础上尽可能地详细地描述。此外，除了代码、特殊名字等内容，你的问题和回答需要是中文
7. **JSON格式输出**: 你的最终输出必须是一个包含问答对列表的JSON对象。格式如下：
   {
     "qa_pairs": [
       {
         "question": "这里是第一个问题",
         "answer": "这里是第一个答案"
       },
       {
         "question": "这里是第二个问题",
         "answer": "这里是第二个答案"
       }
     ]
   }:
Example Output:
   {
     "qa_pairs": [
        {
        "question": "在Gremlin中，dedup()步骤的作用是什么？",
        "answer": "它的作用是去除遍历过程中遇到的重复对象或路径。"
        },
        {
        "question": "在什么场景下使用dedup()特别重要？",
        "answer": "在寻找多度关系（如朋友的朋友）时非常重要，因为可以避免同一个结果被多次返回。例如：g.V().has('person','name','Alice').out('knows').out('knows').dedup().values('name')，这个查询会找到Alice的朋友的朋友，但使用dedup()确保每个人只出现一次，避免了那些通过不同路径到达的重复人员"
        }
     ]
   }

"""

def create_user_prompt(chunk: dict) -> str:
    """根据chunk内容和元数据，动态生成层级化的用户指令。"""
    metadata = chunk.get("metadata", {})
    hierarchy = metadata.get("hierarchy", {})
    
    # 构建目录路径
    path_parts = []
    if hierarchy.get("level_1_title"): path_parts.append(hierarchy["level_1_title"])
    if hierarchy.get("level_2_title"): path_parts.append(hierarchy["level_2_title"])
    if hierarchy.get("level_3_title"): path_parts.append(hierarchy["level_3_title"])
    
    # 去除重复标题（例如当子标题与父标题相同时）
    unique_path_parts = []
    for part in path_parts:
        if not unique_path_parts or unique_path_parts[-1] not in part:
            unique_path_parts.append(part)

    context_path = " > ".join(unique_path_parts)
    content = chunk.get("chunk_content", "")
    
    return f"这是Gremlin权威指南《{context_path}》的内容片段。请根据以下文本生成问答对：\n\n---\n{content}\n---"


client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# @retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=RETRY_WAIT_SECONDS, max=10))
# async def process_chunk(chunk: dict, index: int) -> list:
#     """
#     处理chunk。
#     """
#     user_prompt = create_user_prompt(chunk)
#     print(f"  - 开始处理 Chunk #{index}...")
    
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": user_prompt}
#     ]
    
#     response = await client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=messages,
#         response_format={'type': 'json_object'}
#     )
    
#     response_data = json.loads(response.choices[0].message.content)
#     generated_pairs = response_data.get("qa_pairs", [])
    
#     formatted_pairs = []
#     for pair in generated_pairs:
#         if pair.get("question") and pair.get("answer"):
#             formatted_pairs.append({
#                 "instruction": "你是一位Gremlin知识问答专家，请根据输入的问题，提供准确、简洁的回答。",
#                 "input": pair["question"],
#                 "output": pair["answer"]
#             })
#     print(f"  - 完成 Chunk #{index}，生成 {len(formatted_pairs)} 条QA对。")
#     return formatted_pairs

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=RETRY_WAIT_SECONDS, max=10))
async def process_chunk(chunk: dict, index: int) -> list:
    """
    处理chunk
    """
    user_prompt = create_user_prompt(chunk)
    print(f"  - 开始处理 Chunk #{index}...")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        response_format={'type': 'json_object'}
    )
    

    formatted_pairs = []
    try:
        response_data = json.loads(response.choices[0].message.content)
        generated_pairs = response_data.get("qa_pairs", []) # 即使"qa_pairs"键不存在，也能安全地返回一个空列表
        
        if not generated_pairs:
            print(f"  - 完成 Chunk #{index}，模型判定无相关内容，跳过生成。")
            return [] 

        for pair in generated_pairs:
            if pair.get("question") and pair.get("answer"):
                formatted_pairs.append({
                    "instruction": "你是一位Gremlin知识问答专家，请根据输入的问题，提供准确、简洁的回答。",
                    "input": pair["question"],
                    "output": pair["answer"]
                })
        
        print(f"  - 完成 Chunk #{index}，成功生成 {len(formatted_pairs)} 条QA对。")

    except json.JSONDecodeError:
        print(f"  - 警告: Chunk #{index} 的模型响应不是有效的JSON。跳过。")
        return []
        
    return formatted_pairs


async def main():
    qa_buffer = []
    total_qa_generated = 0
    total_chunks_processed = 0
    
    print(f"开始处理输入文件: {INPUT_FILE}")
    print(f"并发批次大小: {READ_BATCH_SIZE}, 写入文件阈值: {WRITE_BATCH_SIZE}")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            while True:
                batch_lines = [infile.readline() for _ in range(READ_BATCH_SIZE)]
                batch_lines = [line for line in batch_lines if line] # 移除文件末尾的空行
                
                if not batch_lines:
                    break # 文件已读完

                tasks = []
                for line in batch_lines:
                    chunk = json.loads(line)
                    total_chunks_processed += 1
                    tasks.append(process_chunk(chunk, total_chunks_processed))
                
                print(f"\n--- 发起新一批 {len(tasks)} 个并发请求 ---")
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, list):
                        qa_buffer.extend(result)
                    else:
                        print(f"  - 警告: 一个Chunk在 {RETRY_ATTEMPTS} 次重试后仍然处理失败: {result}")
                
                print(f"--- 本批次处理完成。当前缓冲区大小: {len(qa_buffer)} ---")

                if len(qa_buffer) >= WRITE_BATCH_SIZE:
                    print(f"--- 达到写入阈值，将 {len(qa_buffer)} 条数据写入文件... ---")
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
                        for item in qa_buffer:
                            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
                    total_qa_generated += len(qa_buffer)
                    qa_buffer.clear()

    except FileNotFoundError:
        print(f"错误: 输入文件 '{INPUT_FILE}' 未找到。")
        return
        
    if qa_buffer:
        print(f"--- 处理完成，将最后 {len(qa_buffer)} 条数据写入文件... ---")
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
            for item in qa_buffer:
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
        total_qa_generated += len(qa_buffer)
    
    print(f"\n全部完成！总共处理了 {total_chunks_processed} 个Chunks，生成并保存了 {total_qa_generated} 条QA问答对到 '{OUTPUT_FILE}'。")

if __name__ == "__main__":
    asyncio.run(main())