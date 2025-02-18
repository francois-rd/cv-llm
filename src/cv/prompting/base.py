from dataclasses import asdict, dataclass
from typing import Type

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


from ..core import Cluster, ClusterName, ClustersConfig, OutputParser, scrub
from ..segmentation import Transcript


@dataclass
class ClusterPrompt:
    # The name of the cluster.
    name: ClusterName

    # The langchain template for the cluster.
    template: ChatPromptTemplate

    # The output parser to use for this cluster.
    parser: OutputParser


class PromptMaker:
    def __init__(self, cfg: ClustersConfig, parser_type: Type[OutputParser]):
        self.cfg = cfg
        self.parser_type = parser_type

    def __call__(self, transcript: Transcript, *args, **kwargs) -> list[ClusterPrompt]:
        results = []
        for name, cluster in self._get_included_clusters(transcript):
            template = None if len(cluster.lines) == 0 else self._process(cluster)
            data = asdict(cluster.data.parser_data)
            results.append(ClusterPrompt(name, template, self.parser_type(**data)))
        return results

    def _get_included_clusters(
        self,
        transcript: Transcript,
    ) -> list[tuple[ClusterName, Cluster]]:
        included = self.cfg.included_clusters
        return [(k, v) for k, v in transcript.clusters.items() if k in included]

    def _process(self, cluster: Cluster) -> ChatPromptTemplate:
        system = SystemMessagePromptTemplate.from_template(self.cfg.system_prompt)
        template = self.cfg.cluster_template.format(
            cluster_prompt=cluster.data.prompt,
            cluster_text=scrub("\n".join(cluster.lines)),
        )
        human = HumanMessagePromptTemplate.from_template(template)
        return system + human
