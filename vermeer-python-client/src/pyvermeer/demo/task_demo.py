# !/usr/bin/env python3
"""
file: task_demo.py
author: wenyuxuan@baidu.com
"""

from pyvermeer.client.client import PyVermeerClient
from pyvermeer.structure.task_data import TaskCreateRequest


def main():
    """main"""
    client = PyVermeerClient(
        ip="10.41.57.139",
        port=8688,
        token="Q7svB13nYvREB4bDCj7kQnwJEMvLgrgfDimu4h1Fp7CUzQLk758ya1EYwycn1kjbgskiHiKzDni9jEkJcssgTy7rZJdt4gYEkfvjeowZGzSebgiSEU86dgFPXzUUtwrA81vWKm1xfioBcS9GmXjGQoM6C",
        log_level="DEBUG",
    )
    task = client.tasks.get_tasks()

    print(task.to_dict())

    create_response = client.tasks.create_task(
        create_task=TaskCreateRequest(
            task_type='load',
            graph_name='DEFAULT-example',
            params={
                "load.hg_pd_peers": "[\"10.41.57.87:8686\"]",
                "load.hugegraph_name": "DEFAULT/example/g",
                "load.hugegraph_password": "xxx",
                "load.hugegraph_username": "xxx",
                "load.parallel": "10",
                "load.type": "hugegraph"
            },
        )
    )

    print(create_response.to_dict())


if __name__ == "__main__":
    main()
