import requests
import json
import logging

log=logging.getLogger(__name__)
##访问私有服务器api
class API:
    def __init__(self):
        self.token=""
        self.base_url="https://"

    def generate(self,prompt:str)->str:
        try:
            headers_get={
                "Authorization":self.token,
                "Accept":"application/json",
                "User-Agent":"Mozilla/5.0",
                "Origin":"https://",
                "Referer":"https://"
            }
            response=requests.get(self.base_url,headers=headers_get)
            if response.status_code != 200:
                raise ValueError(f"获取chat_id失败:{response.status_code},{response.text}")
            chat_id=response.json()["data"]

            message_url=f"https://{chat_id}"
            payload={
                "message":prompt,
                "re_chat":False,
                "image_list":[],
                "document_list":[],
                "audio_list":[],
                "video_list":[],
                "form_data":{}
            }
            headers_post = {
                "Authorization":self.token,
                "Content-Type":"application/json",
                "Accept":"*/*"
            }
            response=requests.post(
                message_url,
                data=json.dumps(payload),
                headers=headers_post,
                stream=True
            )
            ##将同步流式请求返回字符组合成完整的句子后，再输出
            full_content_list=[]
            for line in response.iter_lines():
                if line and line.startswith(b"data:"):
                    content_json=json.loads(line[5:].decode("utf-8"))
                    full_content_list.append(content_json.get("content",""))
            return "".join(full_content_list)

        except Exception as e:
            log.error(f"API调用失败:{e}")
            return ""