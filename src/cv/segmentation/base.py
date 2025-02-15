from dataclasses import dataclass, field

ClusterName = str
QuestionId = int
Lines = list[str]


@dataclass
class TagsConfig:
    primary_regex: str = r"^(Answered)?\s*Question\s*(.+?)\s*(Ite(ra|ar)tion.+?)?\.\."
    question_group: int = 2
    question_id_regex: str = r"([0-9]+)\s*\w?\s*[0-9]*"


@dataclass
class ClusterData:
    phrasing: str
    questions: list[QuestionId]


@dataclass
class ClustersConfig:
    clusters: dict[ClusterName, ClusterData] = field(default_factory=dict)


@dataclass
class Cluster:
    data: ClusterData
    lines: Lines = field(default_factory=list)


@dataclass
class Transcript:
    clusters: dict[ClusterName, Cluster]
