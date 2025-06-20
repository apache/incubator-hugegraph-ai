#-*- coding: utf-8 -*-
#coding:utf-8
import requests
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def API(question):
    url="https://"#API服务
    headers={
        "Authorization": "",#自己的token
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 Chrome/94.0.4606.71 Safari/537.36",
        "Origin": "https://",
        "Referer" : "https://"
    }
    response = requests.get(url, headers=headers)
    print("状态码：", response.status_code)
    print("响应内容：", response.text)

    my_chat_id = response.json()["data"]
    print("chat_id:", my_chat_id)

    message_url = f"https://{my_chat_id}"
    payload = {
        "message": question,
        "re_chat": False,
        "image_list": [],
        "document_list": [],
        "audio_list": [],
        "video_list": [],
        "form_data": {}
    }
    headers = {
        "Authorization": "",#自己的token
        "Content-Type": "application/json",
        "Accept": "*/*"
    }
    response = requests.post(message_url, data=json.dumps(payload), headers=headers, stream=True)
    #处理流式返回数据
    full_content_list = []
    for line in response.iter_lines():
        if line:
            if line.startswith(b"data:"):
                content_json = json.loads(line[5:].decode("utf-8"))
                full_content_list.append(content_json.get("content", ""))
    full_content = "".join(full_content_list)
    return full_content

def Recall_most_similar_template(query,templates):
    texts=[query]+templates#把query和现有模版组合并到列表
    vectorizer=TfidfVectorizer()
    tfidf_matrix=vectorizer.fit_transform(texts)#向量化
    similarities=cosine_similarity(tfidf_matrix[0:1],tfidf_matrix[1:])#query的TF-IDF向量和模版组的组合，求余弦相似度
    most_similar_idx=similarities.argmax()
    return templates[most_similar_idx]

templates = ['法律知识图谱','人物知识图谱','人物关系图谱','医疗知识图谱','法律知识图谱','企业知识图谱','科研知识图谱','技术图谱','舆情知识图谱','政策知识图谱','法律图谱','地理知识图谱','商业知识图谱','组织知识图谱','知识问答图谱','能力/技能图谱','技术创新图谱','科学知识图谱','金融知识图谱','故事情节图谱','发展脉络图谱']

df = pd.read_excel(r'C:\Data_Test.xlsx')
for index,row in df.iterrows():
    UserInput = row['原文']+row['指令']

    GenerateTemplate4User = f"""用户目标：输入文本和要求，依靠大模型抽取知识图谱。你需要撰写符合用户要求的提示词，用于引导其他大模型根据用户输入文本抽取知识图谱。
    请按照分类和结构化的方式来回答这个问题：
    1. 分类用户提问的主要部分或类别，明确用户要构建什么知识图谱。用户提问可能在句首或句尾，请在这一步明确输出用户提问原句，在原句左右各用五个加号连接+++++；
    2. 详细分析用户目标，并按照用户目标分析图谱的实体分类、关系分类、每个实体的属性；
    3. 记录中间结果，列出实体列表，关系列表，每个实体的属性列表；
    4. 由上述结论总结得出用于指导大模型生成符合用户要求的Prompt，输出最终版本的提示词，该提示词需要包含第二步第三步中详细信息以及Json格式的示例，输出结果放置在下面括号内<<<<<>>>>>

    用户输入如下：
    {UserInput}
    """
    ##回复每个子问题，提取并标记用户要求，据此生成Prompt
    result=API(GenerateTemplate4User)
    df.loc[index,'分析过程']=result

    ##获取+标记的用户要求
    pattern1 = r'\+{3,}(.*?)\+{3,}'
    matches_query=re.findall(pattern1,result,re.DOTALL)
    template=Recall_most_similar_template(matches_query[0],templates)

    ##过去模型生成的Prompt
    pattern2 = r'<{3,}(.*?)>{3,}'
    matches_template=re.findall(pattern2,result,re.DOTALL)
    df.loc[index, '生成指令']=matches_template

    # 测试模型生成的Prompt的建图效果
    BuildGraph4User=f"按照下面要求建立知识图谱：{matches_query[-1]}，然后将输出知识图谱转化为Neo4j风格的可执行语句，用户输入如下{UserInput}。该知识图谱类型有可能与{template}的建立方式接近，也有可能无关，你可以完全忽视结尾给出的参考"
    result=API(BuildGraph4User)
    df.loc[index,'应用该指令效果']=result

    df.to_excel(r'C:\Data_Test.xlsx', index=False)