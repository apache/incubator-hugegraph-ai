import pandas as pd
import json
import os
import random

INPUT_CSV_PATH = 'test_gremlin_qa_dataset.csv' 
OUTPUT_JSON_PATH = 'instruct_data.json'
CHUNK_SIZE = 200  # 每次从CSV读取的行数


TRAIN_RATIO = 0.7  # 训练集与测试集划分比例
TRAIN_OUTPUT_PATH = './train_data/train_dataset.json' # 训练集输出文件
TEST_OUTPUT_PATH = './train_data/test_dataset.json'   # 测试集输出文件

def convert_csv_to_json():

    print("数据转换 CSV -> JSON")
    print(f"输入: '{INPUT_CSV_PATH}'")
    print(f"输出: '{OUTPUT_JSON_PATH}'")
    
    instruction_text = "你是一位精通图数据库查询语言Gremlin的专家。你的任务是根据用户输入的自然语言问题，将其准确地转换为对应的Gremlin查询语句。"
    is_first_object = True

    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            f.write('[\n')
            csv_reader = pd.read_csv(INPUT_CSV_PATH, chunksize=CHUNK_SIZE, iterator=True)
            
            total_rows_processed = 0
            
            for i, chunk_df in enumerate(csv_reader):
                for index, row in chunk_df.iterrows():
                    if pd.notna(row['question']) and pd.notna(row['gremlin_query']):
                        
                        if not is_first_object:
                            f.write(',\n')
                        
                        formatted_data = {
                            "instruction": instruction_text,
                            "input": row['question'],
                            "output": row['gremlin_query']
                        }
                        
                        json_string = json.dumps(formatted_data, ensure_ascii=False, indent=2)
                        f.write(json_string)
                        
                        is_first_object = False
                        total_rows_processed += 1
                
                print(f"  已处理 {i+1} 个数据块，累计处理 {total_rows_processed} 行...")

            f.write('\n]')
            
            print(f"\n 数据转换完成！总共转换了 {total_rows_processed} 条数据,保存文件至 {OUTPUT_JSON_PATH}")
            return True 

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{INPUT_CSV_PATH}'。请检查文件名和路径。")
        return False
    except Exception as e:
        print(f"发生未知错误: {e}")
        return False

def split_and_shuffle_dataset():
    """
    随机打乱，按比例划分训练集和测试集。
    """
    try:
        print(f"  正在从 '{OUTPUT_JSON_PATH}' 加载数据到内存...")
        with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  加载了 {len(data)} 条数据。")

        # 随机打乱数据
        random.shuffle(data)

        # 划分数据集
        split_index = int(len(data) * TRAIN_RATIO) # 计算划分点
        train_data = data[:split_index]
        test_data = data[split_index:]
        
        print(f"  数据划分完毕: {len(train_data)} 条训练数据, {len(test_data)} 条测试数据。")

        print(f"  保存训练集到 '{TRAIN_OUTPUT_PATH}'...")
        with open(TRAIN_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        print(f"  保存测试集到 '{TEST_OUTPUT_PATH}'...")
        with open(TEST_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
            
        print(f"\n 数据集已成功划分为训练集和测试集。")

    except FileNotFoundError:
        print(f"错误: JSON文件 '{OUTPUT_JSON_PATH}' 未找到。请先确保步骤1成功执行。")
    except Exception as e:
        print(f"在划分数据集时发生错误: {e}")


if __name__ == '__main__':
    if convert_csv_to_json():
        split_and_shuffle_dataset()
    else:
        print("\n数据转换失败，停止后续操作。")