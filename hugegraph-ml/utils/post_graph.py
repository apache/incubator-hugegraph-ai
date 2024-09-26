'''
Author: jinsong jsong010123@gmail.com
Date: 2024-09-19 17:13:52
LastEditTime: 2024-09-26 18:09:58
FilePath: /jinsong/code/work/glcc-hugegraph/utils/post_graph.py
Description: 

Copyright (c) 2024 by jinsong, All Rights Reserved. 
'''
"""
Author: jinsong jsong010123@gmail.com
Date: 2024-09-19 17:13:52
LastEditTime: 2024-09-23 12:17:47
FilePath: /jinsong/code/work/glcc-hugegraph/utils/post_graph.py
Description: 

Copyright (c) 2024 by jinsong, All Rights Reserved. 
"""

import requests

def create_graph(url, store, data_path, wal_path):


    payload = f"""
    gremlin.graph=org.apache.hugegraph.HugeFactory
    backend=rocksdb
    serializer=binary
    store={store}
    rocksdb.data_path={data_path}
    rocksdb.wal_path={wal_path}
    """

    headers = {"Content-Type": "text/plain"}

    response = requests.post(url, data=payload, headers=headers)

    print(response.text)

if __name__ == "__main__":
    url = "http://localhost:8080/graphs/hugegraph-diy"
    data_path = "./rks-data-diy"
    wal_path = "./rks-data-diy"
    store = "hugegraph_diy"
    create_graph(url, store, data_path, wal_path)