'''
Author: jinsong jsong010123@gmail.com
Date: 2024-09-19 17:12:09
LastEditTime: 2024-09-26 18:09:37
FilePath: /jinsong/code/work/glcc-hugegraph/utils/delete_graph.py
Description: 

Copyright (c) 2024 by jinsong, All Rights Reserved. 
'''
"""
Author: jinsong jsong010123@gmail.com
Date: 2024-09-19 17:12:09
LastEditTime: 2024-09-23 12:15:47
FilePath: /jinsong/code/work/glcc-hugegraph/utils/delete_graph.py
Description: 

Copyright (c) 2024 by jinsong, All Rights Reserved. 
"""

"""
Author: jinsong jsong010123@gmail.com
Date: 2024-09-19 17:12:09
LastEditTime: 2024-09-23 10:37:23
FilePath: /jinsong/code/work/glcc-hugegraph/delete_graph.py
Description: 

Copyright (c) 2024 by jinsong, All Rights Reserved. 
"""

"""
Author: jinsong jsong010123@gmail.com
Date: 2024-09-19 17:12:09
LastEditTime: 2024-09-19 17:12:13
FilePath: /jinsong/code/work/glcc-hugegraph/delete_graph.py
Description: 

Copyright (c) 2024 by jinsong, All Rights Reserved. 
"""

import requests

def delete_graph(url):
    params = {"confirm_message": "I'm sure to drop the graph"}

    response = requests.delete(url, params=params)

    print(response.text)

if __name__ == "__main__":
    graph = "hugegraph_diy"
    url = f"http://localhost:8080/graphs/{graph}"
    delete_graph(url)