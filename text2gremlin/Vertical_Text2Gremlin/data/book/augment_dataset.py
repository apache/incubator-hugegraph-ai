import json
import os
from dotenv import load_dotenv
import asyncio
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("ds_api_key") 
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"


INPUT_FILE = "gremlin_qa_dataset_final.jsonl"
OUTPUT_FILE = "gremlin_qa_dataset_augmented.jsonl"


READ_BATCH_SIZE = 7     # 并发处理
WRITE_BATCH_SIZE = 100    # 持久化阈值

RETRY_ATTEMPTS = 3 # 重试次数
RETRY_WAIT_SECONDS = 2 #重试等待时间

SYSTEM_PROMPT = """
你是一位精通语言学和Gremlin技术的AI训练数据增强专家。你的任务是根据用户提供的“原始问答对”，生成内容相同、但表述方式不同的多个新版本。

请严格遵守以下规则：
1. **生成多个版本**: 如果原文中没有任何Gremlin相关的信息、问答里没有任何有效信息或者内容明显是错误的的情况下，不要输出任何信息。如果是关于Gremlin并且是有效问答，为每个原始问答对生成3到4个语义完全相同，但措辞不同的新版本。
2. **保持语义一致**: 新问题的意图必须与原问题完全一致，新答案的事实信息必须与原答案完全一致。
3. **风格多样化**: 尝试使用不同的句式、同义词或提问角度。例如，可以将陈述句改为疑问句，或改变语序。
4. **保持专业性**: 无论是问题还是答案，都应保持技术文档的专业和准确风格。
5. **禁止新增或删减信息**: 绝对不能在答案中添加原始答案没有的信息，或删减关键信息。
6. **信息明确完整**:避免指代不明或者模糊不清的信息。比如不要出现"这本书"、“作者”吗，请以"Gremlin官方教程"、"Gremlin官方作者"等代替；不要出现"提供的文本中的..."、"查询示例在提供的文本中给出..."、"提供的问答中..."、"原文中..."等等这类意义不明确的指代。
7. **JSON格式输出**: 你的最终输出必须是一个包含多个泛化版本的JSON对象。格式如下：
   {
     "variations": [
       {
         "new_question": "这是对原始问题的第一个改写版本。",
         "new_answer": "这是对原始答案的第一个改写版本。"
       },
       {
         "new_question": "这是第二个不同的问法。",
         "new_answer": "这是第二个措辞不同的答案。"
       }
     ]
   }
"""

def create_user_prompt(qa_pair: dict) -> str:
    """生成用户指令"""
    original_input = qa_pair.get("input", "")
    original_output = qa_pair.get("output", "")
    return f"请为下面的原始问答对生成3到4个泛化版本。\n\n[原始问题]\n{original_input}\n\n[原始答案]\n{original_output}"


client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# @retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=RETRY_WAIT_SECONDS, max=10))
# async def augment_qa_pair(qa_pair: dict, index: int) -> list:
#     """泛化并格式化输出"""
#     user_prompt = create_user_prompt(qa_pair)
#     print(f"  - 开始泛化 QA #{index}...")

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
#     variations = response_data.get("variations", [])
    
#     augmented_pairs = []
#     augmented_pairs.append(qa_pair) 

#     for variation in variations:
#         if variation.get("new_question") and variation.get("new_answer"):
#             augmented_pairs.append({
#                 "instruction": qa_pair["instruction"],
#                 "input": variation["new_question"],
#                 "output": variation["new_answer"]
#             })
#     print(f"  - 完成 QA #{index}，生成 {len(variations)} 个新版本。")
#     return augmented_pairs

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=RETRY_WAIT_SECONDS, max=10))
async def augment_qa_pair(qa_pair: dict, index: int) -> list:
    """
    泛化QA。
    """
    user_prompt = create_user_prompt(qa_pair)
    print(f"  - 开始泛化 QA #{index}...")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    augmented_pairs = [qa_pair] 

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={'type': 'json_object'}
        )
        
        response_data = json.loads(response.choices[0].message.content)
        variations = response_data.get("variations", [])

        if not variations:
            print(f"  - 完成 QA #{index}，模型未生成泛化版本，仅保留原始数据。")
            return augmented_pairs # 只返回包含原始QA对的列表

        # 添加泛化后的QA对
        newly_generated_count = 0
        for variation in variations:
            if variation.get("new_question") and variation.get("new_answer"):
                augmented_pairs.append({
                    "instruction": qa_pair["instruction"],
                    "input": variation["new_question"],
                    "output": variation["new_answer"]
                })
                newly_generated_count += 1
        
        print(f"  - 完成 QA #{index}，成功生成 {newly_generated_count} 个新版本。")

    except json.JSONDecodeError:
        print(f"  - 警告: QA #{index} 的模型响应不是有效的JSON。")
    except Exception as e:
        print(f"  - 错误: QA #{index} 在调用API时发生最终错误: {e}。")
            
    return augmented_pairs
async def main():
    qa_buffer = []
    total_augmented_pairs = 0
    total_original_pairs_processed = 0
    
    print(f"开始处理输入文件: {INPUT_FILE}")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            while True:
                batch_lines = [infile.readline() for _ in range(READ_BATCH_SIZE)]
                batch_lines = [line for line in batch_lines if line]
                
                if not batch_lines: break

                tasks = []
                for line in batch_lines:
                    qa_pair = json.loads(line)
                    total_original_pairs_processed += 1
                    tasks.append(augment_qa_pair(qa_pair, total_original_pairs_processed))
                
                print(f"\n--- 发起新一批 {len(tasks)} 个并发请求进行泛化 ---")
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, list):
                        qa_buffer.extend(result)
                    else:
                        print(f"  - 警告: 一个QA对在重试后依然泛化失败: {result}")
                
                print(f"--- 本批次处理完成。当前缓冲区大小: {len(qa_buffer)} ---")

                if len(qa_buffer) >= WRITE_BATCH_SIZE:
                    print(f"--- 达到写入阈值，将 {len(qa_buffer)} 条数据写入文件... ---")
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
                        for item in qa_buffer:
                            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
                    total_augmented_pairs += len(qa_buffer)
                    qa_buffer.clear()

    except FileNotFoundError:
        print(f"错误: 输入文件 '{INPUT_FILE}' 未找到。")
        return
        
    if qa_buffer:
        print(f"--- 处理完成，将最后 {len(qa_buffer)} 条数据写入文件... ---")
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
            for item in qa_buffer:
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
        total_augmented_pairs += len(qa_buffer)
    
    print(f"\n 全部完成！总共处理了 {total_original_pairs_processed} 条原始QA对，生成并保存了 {total_augmented_pairs} 条（含原始+泛化）数据到 '{OUTPUT_FILE}'。")

if __name__ == "__main__":
    asyncio.run(main())