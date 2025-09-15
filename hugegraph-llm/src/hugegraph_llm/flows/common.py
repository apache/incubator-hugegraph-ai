from abc import ABC, abstractmethod

from hugegraph_llm.state.ai_state import WkFlowInput


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
