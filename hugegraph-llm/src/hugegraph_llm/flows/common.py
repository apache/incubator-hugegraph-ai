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

from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator

from hugegraph_llm.state.ai_state import WkFlowInput
from hugegraph_llm.utils.log import log


class BaseFlow(ABC):
    """
    Base class for flows, defines three interface methods: prepare, build_flow, and post_deal.
    """

    @abstractmethod
    def prepare(self, prepared_input: WkFlowInput, *args, **kwargs):
        """
        Pre-processing interface.
        """
        pass

    @abstractmethod
    def build_flow(self, *args, **kwargs):
        """
        Interface for building the flow.
        """
        pass

    @abstractmethod
    def post_deal(self, *args, **kwargs):
        """
        Post-processing interface.
        """
        pass

    async def post_deal_stream(
        self, pipeline=None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming post-processing interface.
        Subclasses can override this method as needed.
        """
        flow_name = self.__class__.__name__
        if pipeline is None:
            yield {"error": "No pipeline provided"}
            return
        try:
            state_json = pipeline.getGParamWithNoEmpty("wkflow_state").to_json()
            log.info(f"{flow_name} post processing success")
            stream_flow = state_json.get("stream_generator")
            if stream_flow is None:
                yield {"error": "No stream_generator found in workflow state"}
                return
            async for chunk in stream_flow:
                yield chunk
        except Exception as e:
            log.error(f"{flow_name} post processing failed: {e}")
            yield {"error": f"Post processing failed: {str(e)}"}
