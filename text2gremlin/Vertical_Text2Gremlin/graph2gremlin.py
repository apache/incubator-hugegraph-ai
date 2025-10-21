import pandas as pd
import random
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import sys

load_dotenv()

try:
    client = OpenAI(
        api_key=os.getenv("ds_api_key"),
        base_url=os.getenv("base_url"),
    )
except Exception as e:
    print(f"初始化OpenAI客户端失败，请检查环境变量: {e}")
    sys.exit(1)


COMPONENT_LIBRARY = {}
SCHEMA = {}

def setup_component_library():
    """
    初始化所有数据和素材
    """
    global COMPONENT_LIBRARY, SCHEMA

    SCHEMA = {
        "nodes": {
            "Person": {"properties": ["name", "born"]},
            "Movie": {"properties": ["title", "tagline", "duration"]},
            "Genre": {"properties": ["name"]},
        },
        "edges": {
            "ACTED_IN": {"from": "Person", "to": "Movie"},
            "DIRECTED": {"from": "Person", "to": "Movie"},
            "HAS_GENRE": {"from": "Movie", "to": "Genre"},
        }
    }

    try:
        persons_df = pd.read_csv('/root/lzj/ospp/schema_gremlin/db_data/movie/raw_data/vertex_person.csv', header=1)
        movies_df = pd.read_csv('/root/lzj/ospp/schema_gremlin/db_data/movie/raw_data/vertex_movie.csv', header=1)
        
        persons_df.columns = [col.strip() for col in persons_df.columns]
        movies_df.columns = [col.strip() for col in movies_df.columns]
        
        
        person_name_col = 'name:STRING'
        movie_title_col = 'title:STRING'
        movie_duration_col = 'duration:INT32' 
        
        if person_name_col not in persons_df.columns:
            raise ValueError(f"在 vertex_person.csv 中找不到列: {person_name_col}")
        if movie_title_col not in movies_df.columns:
            raise ValueError(f"在 vertex_movie.csv 中找不到列: {movie_title_col}")
        if movie_duration_col not in movies_df.columns:
            raise ValueError(f"在 vertex_movie.csv 中找不到列: {movie_duration_col}")
        
        person_names = persons_df[person_name_col].dropna().tolist()
        movie_titles = movies_df[movie_title_col].dropna().tolist()
        # 提取电影时长数据
        movie_durations = pd.to_numeric(movies_df[movie_duration_col], errors='coerce').dropna().astype(int).tolist()
        genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller"]

        # 素材库
        COMPONENT_LIBRARY = {
            "nodes": list(SCHEMA["nodes"].keys()),
            "edges": list(SCHEMA["edges"].keys()),
            "properties": SCHEMA["nodes"], 
            "entities": {
                "Person": person_names,
                "Movie": movie_titles,
                "Genre": genres,
                "Duration": movie_durations,
            }
        }
        print("成功从 vertex_person.csv 和 vertex_movie.csv 加载数据素材。")
        return True

    except (FileNotFoundError, ValueError) as e:
        print(f"素材库初始化失败: {e}")
        if 'persons_df' in locals(): print("读取到的 Person 列名:", persons_df.columns.tolist())
        if 'movies_df' in locals(): print("读取到的 Movie 列名:", movies_df.columns.tolist())
        return False


def generate_gremlin_query():
    """随机选择一个查询模式"""
    patterns = [
        find_node_properties, find_one_hop_neighbors, find_one_hop_with_filter,
        count_one_hop_neighbors, find_two_hop_neighbors
    ]
    selected_pattern = random.choice(patterns)
    return selected_pattern()

def find_node_properties():
    """查找某个节点的属性"""
    node_type = random.choice(list(COMPONENT_LIBRARY["nodes"]))
    if not COMPONENT_LIBRARY["entities"].get(node_type): return None, None
    entity_name = random.choice(COMPONENT_LIBRARY["entities"][node_type])
    prop = random.choice(COMPONENT_LIBRARY["properties"][node_type]["properties"])
    id_prop = "name" if node_type in ["Person", "Genre"] else "title"
    entity_name = str(entity_name).replace("'", "\\'")
    query = f"g.V().has('{node_type}', '{id_prop}', '{entity_name}').values('{prop}')"
    description = f"查找实体 '{entity_name}' ({node_type}) 的属性 '{prop}'。"
    return query, description

def find_one_hop_neighbors():
    """一度关联查询"""
    edge = random.choice(list(COMPONENT_LIBRARY["edges"]))
    schema_edge = SCHEMA["edges"][edge]
    start_node, _ = schema_edge["from"], schema_edge["to"]
    if not COMPONENT_LIBRARY["entities"].get(start_node): return None, None
    entity_name = random.choice(COMPONENT_LIBRARY["entities"][start_node])
    id_prop = "name" if start_node in ["Person", "Genre"] else "title"
    entity_name = str(entity_name).replace("'", "\\'")
    query = f"g.V().has('{start_node}', '{id_prop}', '{entity_name}').out('{edge}').elementMap()"
    description = f"查找与 '{entity_name}' ({start_node}) 通过关系 '{edge}' 相连的所有实体。"
    return query, description

def find_one_hop_with_filter():
    """带属性过滤的一度关联查询"""

    # 按电影时长过滤
    if not COMPONENT_LIBRARY["entities"].get("Person") or not COMPONENT_LIBRARY["entities"].get("Duration"): return None, None
    
    entity_name = random.choice(COMPONENT_LIBRARY["entities"]["Person"])

    min_duration = random.choice([90, 120, 150])
    entity_name = str(entity_name).replace("'", "\\'")
    
    # 使用 'duration' 属性进行过滤
    query = f"g.V().has('Person', 'name', '{entity_name}').out('ACTED_IN').has('duration', gt({min_duration})).valueMap('title', 'duration')"
    description = f"查找演员 '{entity_name}' 主演的，时长超过 {min_duration} 分钟的电影及其具体时长。"

    return query, description

def count_one_hop_neighbors():
    """聚合计数查询"""
    edge = random.choice(list(COMPONENT_LIBRARY["edges"]))
    schema_edge = SCHEMA["edges"][edge]
    start_node, _ = schema_edge["from"], schema_edge["to"]
    if not COMPONENT_LIBRARY["entities"].get(start_node): return None, None
    entity_name = random.choice(COMPONENT_LIBRARY["entities"][start_node])
    id_prop = "name" if start_node in ["Person", "Genre"] else "title"
    entity_name = str(entity_name).replace("'", "\\'")
    query = f"g.V().has('{start_node}', '{id_prop}', '{entity_name}').out('{edge}').count()"
    description = f"计算与 '{entity_name}' ({start_node}) 通过关系 '{edge}' 相连的实体数量。"
    return query, description

def find_two_hop_neighbors():
    """二度关联查询"""
    if not COMPONENT_LIBRARY["entities"].get("Person"): return None, None
    entity_name = random.choice(COMPONENT_LIBRARY["entities"]["Person"])
    entity_name = str(entity_name).replace("'", "\\'")
    query = f"g.V().has('Person', 'name', '{entity_name}').out('ACTED_IN').out('HAS_GENRE').path().by('name').by('title').by('name')"
    description = f"查找演员 '{entity_name}' 参演的电影都属于哪些类型？并展示其路径。"
    return query, description

def translate_and_paraphrase(query, description, num_variants=5):
    """Gremlin翻译"""
    if not query: return []
    prompt = f"""
    我正在为一个Text-to-Gremlin项目生成训练数据。请你扮演一个精通图数据库和自然语言的专家。
    # Gremlin 查询:
    {query}
    # 操作描述:
    {description}
    请基于以上信息，生成 {num_variants} 个语义完全相同，但风格、句式多样的自然语言问题。
    请直接返回 {num_variants} 个问题，每行一个，不要添加任何序号、标题或解释。
    """
    try:
        print("调用API生成问题...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.3
        )
        questions = response.choices[0].message.content.strip().split('\n')
        qa_pairs = [{"question": q.lstrip("*-123456789. "), "gremlin_query": query} for q in questions if q]
        return qa_pairs
    except Exception as e:
        print(f"调用API时出错: {e}")
        return []

def main(num_to_generate=500, questions_per_query=3):
    if not setup_component_library():
        print("已终止。")
        return

    num_queries_needed = (num_to_generate // questions_per_query) + 1
    all_qa_pairs = []
    generated_queries = set()

    print(f"\n目标生成 {num_to_generate} 条 QA 对，启动生成器引擎...")
    
    while len(all_qa_pairs) < num_to_generate:
        gremlin_query, description = generate_gremlin_query()
        if gremlin_query and gremlin_query not in generated_queries:
            generated_queries.add(gremlin_query)
            print(f"\n生成新 Gremlin 查询 ({len(generated_queries)}/{num_queries_needed}): {gremlin_query}")
            qa_pairs = translate_and_paraphrase(gremlin_query, description, questions_per_query)
            if qa_pairs:
                all_qa_pairs.extend(qa_pairs)
                print(f" -> 成功泛化 {len(qa_pairs)} 个问题。当前总数: {len(all_qa_pairs)}")
        if len(generated_queries) > num_queries_needed * 2 and len(all_qa_pairs) < num_to_generate:
             print("警告：已生成的查询数量远超所需，可能素材库已难以产生新查询。提前终止。")
             break
    
    print(f"\n--- 生成完成！共获得 {len(all_qa_pairs)} 条 QA 对 ---")
    output_df = pd.DataFrame(all_qa_pairs)
    output_df.to_csv("test_gremlin_qa_dataset.csv", index=False, encoding='utf-8-sig')
    print("数据集已保存到 gremlin_qa_dataset.csv")

if __name__ == "__main__":
    main(num_to_generate=3000, questions_per_query=5)