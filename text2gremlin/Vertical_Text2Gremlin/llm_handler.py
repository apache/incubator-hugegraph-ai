# LLM交互模块，泛化qa数据
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict



load_dotenv()
api_key = os.environ.get("DEEPSEEK_API_KEY")
client = OpenAI(
    api_key=os.getenv("ds_api_key"),
    base_url="https://api.deepseek.com/v1",
)
client = OpenAI(
    api_key=os.getenv("ds_api_key"),
    base_url="https://api.deepseek.com/v1",
)

def generate_gremlin_variations(seed_query: str, seed_questions: List[str]) -> List[str]:
    """
    基于种子数据，调用LLM生成多个Gremlin查询变体。
    """
    system_prompt = """
    你是一位精通图数据库和 Gremlin 查询语言的专家。
    你的任务是基于用户提供的种子Gremlin查询和相关的一组自然语言问题,生成多个新的、有意义的、但语法结构或查询参数不同的Gremlin查询,你新生成的语句必须与原有的语句意思作用不同，但要保证gremlin查询语句的正确性，要符合gremlin的语法定义。
    请确保你的输出是严格的JSON格式。

    EXAMPLE INPUT:
    - Seed Query: "g.V().has('Person', 'name', 'James Earl Jones').out('ACTED_IN').has('duration', gt(150)).valueMap('title', 'duration')"
    - Seed Questions: ["哪些由 James Earl Jones 出演的电影时长超过了 150 分钟...?", "查询 James Earl Jones 主演的电影中，哪些片长超过两个半小时...?"]
    
    EXAMPLE JSON OUTPUT:
    {
        "gremlin_variations": [
            "g.V().has('Person', 'name', 'James Earl Jones').out('ACTED_IN').has('duration', gt(150)).count()",
            "g.V().has('Person', 'name', 'James Earl Jones').out('DIRECTED').has('duration', gt(120)).valueMap('title')",
            "g.V().has('Person', 'name', 'Frank Whaley').out('ACTED_IN').has('duration', gt(150)).valueMap('title', 'duration')"
        ]
    }
    """
    
    user_prompt = f"""
    请基于以下种子数据,为我生成3-5个Gremlin查询变体,变体必须与原有的语句意思作用不同,要有差异区别，比如场景与类型可以多样化,但要保证gremlin查询语句的正确性,要符合gremlin的语法定义。

    种子问题列表,以下几个问题都对应同一条gremlin查询语句:
    {json.dumps(seed_questions, ensure_ascii=False)}

    上面几个种子问题都对应下面这一条种子Gremlin查询:
    "{seed_query}"
    """
    # print(f"\n gremlin生成的sys prompt: \n{system_prompt}")
    # print(f"\n gremlin生成的user prompt: \n{user_prompt}")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={'type': 'json_object'},
            temperature=1.3 
        )
        content = response.choices[0].message.content
        return json.loads(content).get("gremlin_variations", [])
    except Exception as e:
        print(f"Error calling LLM for Gremlin generation: {e}")
        return []

def generate_texts_for_gremlin(gremlin_query: str) -> List[str]:
    """
    为一条合法的Gremlin查询，生成多个对应的、多样化的自然语言问题。
    """
    system_prompt = """
    你是一位精通Gremlin查询和自然语言的专家。
    你的任务是为给定的Gremlin查询，生成多个语义完全相同但表述方式不同的自然语言问题。
    请确保你的输出是严格的JSON格式。

    EXAMPLE INPUT:
    - Gremlin Query: "g.V().has('Person', 'name', 'James Earl Jones').out('ACTED_IN').has('duration', lt(90)).count()"

    EXAMPLE JSON OUTPUT:
    {
        "questions": [
            "James Earl Jones主演的电影中，有多少部片长少于90分钟？",
            "统计一下James Earl Jones参演的影片里，时长在一个半小时以内的作品总数。",
            "查询时长低于90分钟的、由James Earl Jones出演的电影数量。"
        ]
    }
    """

    user_prompt = f"""
    请为下面这条Gremlin查询，生成3-5个不同的、但意思完全一样的自然语言问题,且问题必须严格符合Gremlin查询语义。表述尽量符合人的语言表述习惯，几条自然语言问题之间要不同，差异尽量大一点，比如长度可以长一点:

    "{gremlin_query}"
    """
    # print(f"\n 问题生成的sys prompt: \n{system_prompt}")
    # print(f"\n 问题生成的user prompt: \n{user_prompt}")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1.4,
            response_format={'type': 'json_object'}
        )
        content = response.choices[0].message.content
        return json.loads(content).get("questions", [])
    except Exception as e:
        print(f"Error calling LLM for text generation: {e}")
        return []