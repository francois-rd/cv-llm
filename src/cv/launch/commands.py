from enum import Enum
import sys
import os

import coma

from .base import Configs as Cfgs, init
from ..consolidate import ConsolidateConfig, Consolidator
from ..core import ClustersConfig, DefaultScoreParser, enum_from_str
from ..extract import Extract, ClusterOutput
from ..llms import LLMsConfig, TransformersConfig
from ..segmentation import ConvertTagsToTranscript, TagsConfig, Tagger, Transcript
from ..io import (
    PathConfig,
    walk_dataclass_json,
    load_dataclass_jsonl,
    save_json,
    save_dataclass_json,
    save_dataclass_jsonl,
    walk_docx,
    walk_json,
)


def docx_to_json(paths: PathConfig):
    for lines, walk in walk_docx(paths.docx_transcript_dir):
        # Extract the 'assign ID' to create the output filename.
        a_id = walk.base.split("_")[0]
        output_file = str(os.path.join(paths.json_transcript_dir, a_id + ".json"))
        save_json(output_file, lines, indent=4)


def segment(paths: PathConfig, tags: TagsConfig, clusters: ClustersConfig):
    tag = Tagger(tags)
    to_transcript = ConvertTagsToTranscript(clusters)
    for lines, walk in walk_json(paths.json_transcript_dir):
        transcript = to_transcript(lines, tag(lines))
        output_file = walk.map(paths.clustered_transcript_dir)
        save_dataclass_json(output_file, transcript, indent=4)


class RerunProtocol(Enum):
    """The protocol for treating existing output files during a rerun."""

    NEVER = "NEVER"  # Never allow rerun. Raise an error if previous files exist.
    MISSING = "MISSING"  # Allow a partial rerun. Only run missing files.
    OVERWRITE = "OVERWRITE"  # Allow a full rerun. Overwrite every file.


def extract(
    paths: PathConfig,
    clusters: ClustersConfig,
    llms: LLMsConfig,
    transformers_cfg: TransformersConfig,
    rerun_protocol: RerunProtocol,
):
    do_extract = Extract(
        clusters,
        llms,
        DefaultScoreParser,
        transformers_cfg=transformers_cfg,
    )
    root = paths.clustered_transcript_dir
    for transcript, walk in walk_dataclass_json(root, t=Transcript):
        # Manipulate file paths.
        a_id = os.path.splitext(walk.base)[0]
        output_file = f"{paths.run_dir}/{llms.llm}/{a_id}.jsonl"
        if os.path.exists(output_file):
            if rerun_protocol == RerunProtocol.NEVER:
                raise ValueError(
                    f"RerunProtocol set to '{rerun_protocol}' "
                    f"but file exists: {output_file}"
                )
            elif rerun_protocol == RerunProtocol.MISSING:
                continue
            elif rerun_protocol == RerunProtocol.OVERWRITE:
                pass
            else:
                raise ValueError(f"Unsupported RerunProtocol: {rerun_protocol}")

        # Use an LLM to extract data from the transcript.
        save_dataclass_jsonl(output_file, *do_extract(transcript))


class Consolidate:
    def __init__(
        self,
        paths: PathConfig,
        clusters: ClustersConfig,
        consolidate: ConsolidateConfig,
    ):
        self.paths = paths
        self.consolidate = Consolidator(
            consolidate,
            clusters,
            self.get_assign_id,
            self.get_run_id_and_llm,
            self.load_data,
        )

    def run(self):
        self.consolidate(self.paths.raw_scores_dir, self.paths.consolidate_file)

    @staticmethod
    def get_assign_id(filename: str) -> str:
        return os.path.splitext(os.path.basename(filename))[0]

    def get_run_id_and_llm(self, path: str) -> tuple[str, str]:
        relpath = os.path.relpath(path, start=self.paths.raw_scores_dir)
        run_id = self._get_run_id(relpath)
        return run_id, os.path.relpath(relpath, start=run_id)

    @staticmethod
    def _get_run_id(path: str) -> str:
        top_level, current_level = None, path
        while True:
            current_level = os.path.dirname(current_level)
            if current_level == "":
                break
            else:
                top_level = current_level
        if top_level is None:
            raise ValueError
        return str(top_level)

    @staticmethod
    def load_data(file_path: str) -> list[ClusterOutput]:
        return load_dataclass_jsonl(file_path, t=ClusterOutput)


def register():
    """Registers all known commands with Coma."""
    coma.register("test.launch", lambda: print("Successfully launched."))
    coma.register("docx.to.json", docx_to_json, **Cfgs.add(Cfgs.paths))
    coma.register(
        "segment",
        segment,
        **Cfgs.add(Cfgs.paths, Cfgs.tags, Cfgs.clusters),
    )

    @coma.hooks.hook
    def extract_pre_init_hook(known_args, configs):
        protocol = enum_from_str(RerunProtocol, known_args.rerun_protocol)
        configs["rerun_protocol"] = protocol

    coma.register(
        "extract",
        extract,
        parser_hook=coma.hooks.parser_hook.factory(
            "-r",
            "--rerun-protocol",
            default="never",
            choices=["never", "missing", "overwrite"],
            help="set the protocol for treating existing output files during rerun",
        ),
        pre_init_hook=extract_pre_init_hook,
        **Cfgs.add(Cfgs.paths, Cfgs.clusters, Cfgs.llms, Cfgs.transformers),
    )

    coma.register(
        "consolidate",
        Consolidate,
        **Cfgs.add(Cfgs.paths, Cfgs.clusters, Cfgs.consolidate),
    )


def launch():
    """Launches the application with Coma."""
    init()
    register()
    try:
        coma.wake()
    except AttributeError:
        if len(sys.argv) == 1:
            os.chdir(os.environ["DEFAULT_CONFIG_DIR"])
            coma.wake(args=[os.environ["DEFAULT_COMMAND"]])
        else:
            raise
