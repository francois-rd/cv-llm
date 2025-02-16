from dataclasses import dataclass, field

ClusterName = str
Lines = list[str]
Prompt = str
QuestionId = int


@dataclass
class ParserData:
    min_score: float = 0.0
    max_score: float = 1.0
    force_int: bool = False
    int_tol: float = 0.0001


@dataclass
class ClusterData:
    prompt: Prompt
    questions: list[QuestionId]
    parser_data: ParserData = field(default_factory=ParserData)


@dataclass
class ClustersConfig:
    system_prompt: Prompt
    cluster_template: Prompt = "{cluster_prompt}\nBe concise.\n\n{cluster_text}"
    clusters: dict[ClusterName, ClusterData] = field(default_factory=dict)


@dataclass
class Cluster:
    data: ClusterData
    lines: Lines = field(default_factory=list)
