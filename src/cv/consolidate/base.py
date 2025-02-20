from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from ..core import ClusterName, ClustersConfig
from ..extract import ClusterOutput
from ..io import ensure_path, walk_fn
from ..llms import Nickname


@dataclass
class ConsolidateConfig:
    assign_id_column_name: str = "assign_id"
    llm_column_name: str = "llm"
    ordered_run_ids: list[str] = field(default_factory=list)
    llms: list[Nickname] = field(default_factory=list)
    assign_id_blacklist: list[str] = field(default_factory=list)


@dataclass
class _IntermediaryResult:
    run_id: str
    llm: str
    data: dict[ClusterName, ClusterOutput]


@dataclass
class _CleanResult:
    llm: str
    data: dict[ClusterName, Any]


class Consolidator:
    def __init__(
        self,
        cfg: ConsolidateConfig,
        clusters_cfg: ClustersConfig,
        assign_id_extractor: Callable[[str], str],
        run_id_and_llm_extractor: Callable[[str], tuple[str, str]],
        data_loader: Callable[[str], list[ClusterOutput]],
    ):
        self.cfg = cfg
        self.assign_id = assign_id_extractor
        self.run_id_and_llm = run_id_and_llm_extractor
        self.load = data_loader
        self.reversed_run_ids = list(reversed(cfg.ordered_run_ids))
        self.all_cluster_names = list(clusters_cfg.clusters.keys())

    def __call__(self, root_dir: str, output_file: str, *args, **kwargs):
        result_by_aid = self._get_intermediary_results(root_dir)
        result_by_aid = {k: self._keep_only_latest(v) for k, v in result_by_aid.items()}
        df = self._to_df(result_by_aid)
        df.to_csv(ensure_path(output_file), index=False)

    def _get_intermediary_results(
        self,
        root_dir: str,
    ) -> dict[str, dict[str, _IntermediaryResult]]:
        result_by_aid = {}
        for data, walk in walk_fn(root_dir, self.load):
            a_id = self.assign_id(walk.base)
            if a_id in self.cfg.assign_id_blacklist:
                continue
            run_id, llm = self.run_id_and_llm(walk.root)
            if run_id not in self.cfg.ordered_run_ids or llm not in self.cfg.llms:
                continue
            data = {c.cluster_name: c for c in data}
            result = _IntermediaryResult(run_id, llm, data)
            result_by_aid.setdefault(a_id, {})[run_id] = result
        return result_by_aid

    def _keep_only_latest(
        self,
        aid_result: dict[str, _IntermediaryResult],
    ) -> _CleanResult:
        latest = {}
        for name in self.all_cluster_names:
            if name in latest:
                continue
            for run_id in self.reversed_run_ids:
                if run_id in aid_result:
                    intermediary = aid_result[run_id]
                    if name in intermediary.data:
                        cluster_output = intermediary.data[name]
                        latest[name] = cluster_output.llm_output
                        # This break is important to ensure that we keep data
                        # from only the latest run ID that matches.
                        break
            if name not in latest:
                # If data is entirely missing, mark is as not found.
                latest[name] = None

        # Since all _IntermediaryResult in the input have the same LLM,
        # arbitrarily pick the first one.
        llm = next(aid_result.items().__iter__())[1].llm
        return _CleanResult(llm, latest)

    def _to_df(self, clean_results: dict[str, _CleanResult]) -> pd.DataFrame:
        data = {}
        for a_id, result in clean_results.items():
            data.setdefault(self.cfg.assign_id_column_name, []).append(a_id)
            data.setdefault(self.cfg.llm_column_name, []).append(result.llm)
            for name, value in result.data.items():
                data.setdefault(name, []).append(value)
        return pd.DataFrame(data)
