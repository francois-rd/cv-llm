from dataclasses import dataclass

from ..core import Cluster, ClusterName


@dataclass
class TagsConfig:
    primary_regex: str = r"^(Answered)?\s*Question\s*(.+?)\s*(Ite(ra|ar)tion.+?)?\.\."
    question_group: int = 2
    question_id_regex: str = r"([0-9]+)\s*\w?\s*[0-9]*"


@dataclass
class Transcript:
    clusters: dict[ClusterName, Cluster]
