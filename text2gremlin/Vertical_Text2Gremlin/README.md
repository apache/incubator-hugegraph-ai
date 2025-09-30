## 文件介绍

- **`./graph2gremlin.py`**：初始基于模板和graph数据生成gremlin数据，由模板保证正确性，并对gremlin数据进行翻译与初步question泛化

- **`./gremlin_checker.py`**：使用Antlr4进行语法检查

- **`./llm_handler.py`**：LLM交互模型，每次输入种子数字的每个批次的qa数据（种子数据生成时会对query进行一次小批量泛化），让LLM了解如何text2gremlin怎么写，先泛化gemlin，再翻译、泛化query

- **`./qa_generalize.py`**：调用gremlin_checker、llm_handler进行种子数据泛化

- **`./instruct_convert.py`**：指令格式转换、训练集测试集划分

- **`./da_data`**：schema与graph数据

- **`./data/seed_data`**：种子数据 (待上传)

- **`./data/vertical_training_sets`**：垂类场景泛化数据 (待上传)
