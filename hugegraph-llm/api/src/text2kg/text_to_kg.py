import os


from api.src.llm.openai import OpenAIChat
from text2kg.data_to_data import DataDisambiguation
from text2kg.data_to_kg import DataToKg
from text2kg.text_to_data import TextToData


class TextToKg:
    def __init__(self, name):
        self.name = name
        self.text2kg = []

    def text2data(self, llm):
        self.text2kg.append(TextToData(llm=llm))
        return self

    def data2data(self, llm):
        self.text2kg.append(DataDisambiguation(llm=llm))
        return self

    def data2kg(self):
        self.text2kg.append(DataToKg())
        return self

    def run(self, result):
        for i in self.text2kg:
            result = i.run(result)



if __name__ == '__main__':
    # need your proxy
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    # need your api_key
    api_key = ''

    default_llm = OpenAIChat(
        openai_api_key=api_key, model_name="gpt-3.5-turbo-16k", max_tokens=4000
    )
    text = (
        "Meet Sarah, a 30-year-old attorney, and her roommate, James, whom she's shared a home with since 2010. James, "
        "in his professional life, works as a journalist. Additionally, Sarah is the proud owner of the website "
        "www.sarahsplace.com, while James manages his own webpage, though the specific URL is not mentioned here. "
        "These two individuals, Sarah and James, have not only forged a strong personal bond as roommates but have "
        "also carved out their distinctive digital presence through their respective webpages, showcasing their "
        "varied interests and experiences.")
    my_schema = "Schema: Nodes: [Person {age: integer, name: string}] Relationships: [Person, roommate, Person]"
    ops = TextToKg(name="1")
    ops.text2data(llm=default_llm).data2data(llm=default_llm).data2kg().run(text)
