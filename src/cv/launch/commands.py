import sys
import os

import coma

from .base import Configs as Cfgs, init
from ..core import ClustersConfig, DefaultScoreParser
from ..extract import Extract
from ..llms import LLMsConfig
from ..segmentation import ConvertTagsToTranscript, TagsConfig, Tagger, Transcript
from ..io import (
    PathConfig,
    load_dataclass_json,
    load_docx,
    load_json,
    save_json,
    save_dataclass_json,
    save_dataclass_jsonl,
)


def docx_to_json(path: PathConfig):
    for root, _, files in os.walk(path.docx_transcript_dir):
        for filename in files:
            # Extract the 'assign ID' to create the output filename.
            a_id = filename.split("_")[0]
            output_file = str(os.path.join(path.json_transcript_dir, a_id + ".json"))

            # Convert the data and write to file in JSON format.
            lines = load_docx(str(os.path.join(root, filename)))
            save_json(output_file, lines, indent=4)


def segment(path: PathConfig, tags: TagsConfig, clusters: ClustersConfig):
    tag = Tagger(tags)
    to_transcript = ConvertTagsToTranscript(clusters)
    for root, _, files in os.walk(path.json_transcript_dir):
        for filename in files:
            lines = load_json(str(os.path.join(root, filename)))
            output_file = str(os.path.join(path.clustered_transcript_dir, filename))
            transcript = to_transcript(lines, tag(lines))
            save_dataclass_json(output_file, transcript, indent=4)


def extract(path: PathConfig, clusters: ClustersConfig, llms: LLMsConfig):
    do_extract = Extract(clusters, llms, DefaultScoreParser)
    for root, _, files in os.walk(path.clustered_transcript_dir):
        for filename in files:
            # Manipulate file paths.
            file_path = str(os.path.join(root, filename))
            a_id = os.path.splitext(os.path.basename(filename))[0]
            output_file = f"{path.raw_scores_dir}/{llms.llm}/{a_id}.jsonl"

            # Use an LLM to extract data from the transcript.
            transcript = load_dataclass_json(file_path, t=Transcript)
            save_dataclass_jsonl(output_file, *do_extract(transcript))


def register():
    """Registers all known commands with Coma."""
    coma.register("test.launch", lambda: print("Successfully launched."))
    coma.register("docx.to.json", docx_to_json, **Cfgs.add(Cfgs.paths))
    coma.register(
        "segment",
        segment,
        **Cfgs.add(Cfgs.paths, Cfgs.tags, Cfgs.clusters),
    )
    coma.register(
        "extract",
        extract,
        **Cfgs.add(Cfgs.paths, Cfgs.clusters, Cfgs.llms),
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
