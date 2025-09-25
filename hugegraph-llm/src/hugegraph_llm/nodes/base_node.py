#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from PyCGraph import GNode, CStatus
from hugegraph_llm.nodes.util import init_context
from hugegraph_llm.state.ai_state import WkFlowInput, WkFlowState


class BaseNode(GNode):
    context: WkFlowState = None
    wk_input: WkFlowInput = None

    def init(self):
        return init_context(self)

    def node_init(self):
        """
        Node initialization method, can be overridden by subclasses.
        Returns a CStatus object indicating whether initialization succeeded.
        """
        return CStatus()

    def run(self):
        """
        Main logic for node execution, can be overridden by subclasses.
        Returns a CStatus object indicating whether execution succeeded.
        """
        sts = self.node_init()
        if sts.isErr():
            return sts
        self.context.lock()
        try:
            data_json = self.context.to_json()
        finally:
            self.context.unlock()

        try:
            res = self.operator_schedule(data_json)
        except Exception as exc:
            import traceback

            node_info = f"Node type: {type(self).__name__}, Node object: {self}"
            err_msg = f"Node failed: {exc}\n{node_info}\n{traceback.format_exc()}"
            return CStatus(-1, err_msg)

        self.context.lock()
        try:
            if isinstance(res, dict):
                self.context.assign_from_json(res)
        finally:
            self.context.unlock()
        return CStatus()

    def operator_schedule(self, data_json):
        """
        Interface for scheduling the operator, can be overridden by subclasses.
        Returns a CStatus object indicating whether scheduling succeeded.
        """
        pass
