from .base import LLM, LLMOutput, Nickname
from ..prompting import ClusterPrompt


class TransformersLLM(LLM):
    def __init__(self, nickname: Nickname, *args, **kwargs):
        super().__init__(nickname, *args, **kwargs)

    def invoke(self, prompt: ClusterPrompt, *args, **kwargs) -> LLMOutput:
        raise NotImplementedError
