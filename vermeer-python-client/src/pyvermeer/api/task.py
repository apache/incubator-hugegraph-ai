# !/usr/bin/env python3
"""
file: task.py
author: wenyuxuan@baidu.com
"""

from pyvermeer.api.base import BaseModule

from pyvermeer.structure.task_data import TasksResponse, TaskCreateRequest, TaskCreateResponse, TaskResponse


class TaskModule(BaseModule):
    """Task"""

    def get_tasks(self) -> TasksResponse:
        """获取任务列表"""
        response = self._send_request(
            "GET",
            "/tasks"
        )
        return TasksResponse(response)

    def get_task(self, task_id: int) -> TaskResponse:
        """获取单个任务信息"""
        response = self._send_request(
            "GET",
            f"/task/{task_id}"
        )
        return TaskResponse(response)

    def create_task(self, create_task: TaskCreateRequest) -> TaskCreateResponse:
        """创建新任务"""
        response = self._send_request(
            method="POST",
            endpoint="/tasks/create",
            params=create_task.to_dict()
        )
        return TaskCreateResponse(response)
