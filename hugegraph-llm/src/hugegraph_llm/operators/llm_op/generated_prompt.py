import re
import pandas as pd
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from API_test import API

class PromptGenerator:
    def __init__(self,llm:BaseLLM,templates:List[str]):
        self.llm=llm
        self.templates=templates###知识图谱类型的模版表（可以不要这个模版表，效果也好）
        self.template_labels=list(set(templates))
        self.NEEDLE_PATTERN=r'\+{3,}(.*?)\+{3,}'###匹配用户需要生成的知识图谱类型，用于检索最相近的模版
        self.PROMPT_PATTERN=r'<{3,}(.*?)>{3,}'###匹配生成的模版

    ###结合用户指令，生成prompt
    def run(self,user_input:str)->str:
            ###分析用户输入内容与指令，生成分析与模版
            generation_prompt=self._format_analysis_prompt(user_input)
            analysis_result=self.llm.generate(prompt=generation_prompt)

            ###图谱类型匹配(可选)
            extracted_instruction=self._extract_text_block(analysis_result,self.NEEDLE_PATTERN)
            matched_template=self._recall_most_similar_template(extracted_instruction,self.template_labels)

            ###提取得到生成的Prompt模版
            generated_prompt=self._extract_text_block(analysis_result,self.PROMPT_PATTERN)

            #生成最终模版
            final_prompt=self._format_build_graph_prompt(user_input,generated_prompt,matched_template)

            return final_prompt

    ###TF_IDF召回最接近的模版(可选)
    def _recall_most_similar_template(self,query:str,templates:List[str])->str:
        texts=[query]+templates
        vectorizer=TfidfVectorizer()
        tfidf_matrix=vectorizer.fit_transform(texts)
        similarities=cosine_similarity(tfidf_matrix[0:1],tfidf_matrix[1:])
        most_similar_idx=similarities.argmax()
        return templates[most_similar_idx]

    ###从生成的分析内容中提取出模版
    def _extract_text_block(self,text:str,pattern:str)->str:
        matches=re.findall(pattern,text,re.DOTALL)
        if matches:
            return matches[0].strip()
        return ""

    ###构建模版
    def _format_analysis_prompt(self,user_input:str)->str:
        return f"""用户目标：输入文本和要求，依靠大模型抽取知识图谱。你需要撰写符合用户要求的提示词，用于引导其他大模型根据用户输入文本抽取知识图谱。
                    请按照分类和结构化的方式来回答这个问题：
                    1. 分类用户提问的主要部分或类别，明确用户要构建什么知识图谱。用户提问可能在句首或句尾，请在这一步明确输出用户提问原句，在原句左右各用五个加号连接+++++；
                    2. 详细分析用户目标，并按照用户目标分析图谱的实体分类、关系分类、每个实体的属性；
                    3. 记录中间结果，列出实体列表，关系列表，每个实体的属性列表；
                    4. 由上述结论总结得出用于指导大模型生成符合用户要求的Prompt，输出最终版本的提示词，该提示词需要包含第二步第三步中详细信息以及Json格式的示例，输出结果放置在下面括号内<<<<<>>>>>

                    用户输入如下：
                    {user_input}
                    """
    ###用于测试
    def _format_build_graph_prompt(self,user_input:str,instruction:str,template:str)->str:
        return (
            f"按照下面要求建立知识图谱：{instruction}，然后将输出知识图谱转化为Neo4j风格的可执行语句。"
            f"用户输入如下：{user_input}。\n该知识图谱类型有可能与“{template}”的建立方式接近，也有可能无关，你可以完全忽视结尾给出的参考。"
        )

###用于测试，实际流程中无需保存入excel
if __name__=="__main__":
    user_input=""
    llm=API()
    templates=[
        '法律知识图谱','人物知识图谱','人物关系图谱','医疗知识图谱','企业知识图谱','科研知识图谱','技术图谱','舆情知识图谱','政策知识图谱','法律图谱','地理知识图谱','商业知识图谱',
        '组织知识图谱','知识问答图谱','能力/技能图谱','技术创新图谱','科学知识图谱','金融知识图谱','故事情节图谱','发展脉络图谱'
    ]

    generator=PromptGenerator(llm=llm,templates=templates)
    result_context=generator.run(user_input)
