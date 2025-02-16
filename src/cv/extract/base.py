from dataclasses import dataclass
from typing import Optional, Type

from langchain_core.runnables import Runnable, chain
from langchain_core.runnables.base import RunnableEach

from ..core import ClusterName, ClustersConfig, ScoreOutputParser
from ..llms import LLMsConfig, load_llm
from ..prompting import ClusterPrompt, PromptMaker
from ..segmentation import Transcript


@dataclass
class ClusterOutput:
    cluster_name: ClusterName
    llm_score: Optional[float]
    error_message: Optional[str]


class Extract:
    def __init__(
        self,
        clusters: ClustersConfig,
        llms: LLMsConfig,
        parser_type: Type[ScoreOutputParser],
        *args,
        **kwargs,
    ):
        self.make_prompts = PromptMaker(clusters, parser_type)
        self.llm = load_llm(llms, *args, **kwargs)
        self.chain = self._generate_chain()

    def __call__(self, transcript: Transcript, *args, **kwargs) -> list[ClusterOutput]:
        return self.chain.invoke(transcript)

    def _generate_preprocessor(self) -> Runnable:
        @chain
        def runnable(transcript: Transcript):
            return self.make_prompts(transcript)

        return runnable

    def _generate_llm(self) -> Runnable:
        @chain
        def runnable(p: ClusterPrompt):
            if p.template is None:
                # If there is no cluster data for whatever reason (parsing error or
                # legitimately missing data from the transcript), skip the LLM.
                return ClusterOutput(p.name, None, "No cluster data.")
            output = self.llm.invoke(p)
            return ClusterOutput(
                cluster_name=p.name,
                llm_score=p.parser(output.generated_text),
                error_message=output.error_message,
            )

        return runnable

    def _generate_chain(self) -> Runnable:
        return self._generate_preprocessor() | RunnableEach(bound=self._generate_llm())
