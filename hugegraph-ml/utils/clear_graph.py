'''
Author: jinsong jsong010123@gmail.com
Date: 2024-09-23 12:20:21
LastEditTime: 2024-09-26 16:58:14
FilePath: /jinsong/code/work/glcc-hugegraph/utils/clear_graph.py
Description: 

Copyright (c) 2024 by jinsong, All Rights Reserved. 
'''
"""
Author: jinsong jsong010123@gmail.com
Date: 2024-09-23 12:20:21
LastEditTime: 2024-09-23 13:52:31
FilePath: /jinsong/code/work/glcc-hugegraph/utils/clear_graph.py
Description: 

Copyright (c) 2024 by jinsong, All Rights Reserved. 
"""


import requests

def clear_graph(url):
    params = {"confirm_message": "I'm sure to delete all data"}

    response = requests.delete(url, params=params)

    print(response.text)

if __name__ == "__main__":
    graph = "hugegraph_diy"
    url = f"http://localhost:8080/graphs/{graph}/clear"
    clear_graph(url)