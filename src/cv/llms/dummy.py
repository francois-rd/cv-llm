from .base import LLM, LLMOutput
from ..prompting import ClusterPrompt


class DummyLLM(LLM):
    def invoke(self, prompt: ClusterPrompt, *args, **kwargs) -> LLMOutput:
        return LLMOutput(str(1.0), None)
