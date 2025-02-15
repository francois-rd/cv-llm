import sys
import os

import coma

from .base import Configs as Cfgs, init
from ..io import PathConfig, load_docx, load_json, save_json, save_dataclass_json
from ..segmentation import ClustersConfig, ConvertTagsToTranscript, TagsConfig, Tagger


def docx_to_json(path: PathConfig):
    for root, _, files in os.walk(path.docx_transcript_dir):
        for filename in files:
            # Extract the 'assign ID' to create the output filename.
            aid = filename.split("_")[0]
            output_file = str(os.path.join(path.json_transcript_dir, aid + ".json"))

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


def register():
    """Registers all known commands with Coma."""
    coma.register("test.launch", lambda: print("Successfully launched."))
    coma.register("docx.to.json", docx_to_json, **Cfgs.add(Cfgs.paths))
    coma.register(
        "segment",
        segment,
        **Cfgs.add(Cfgs.paths, Cfgs.tags, Cfgs.clusters),
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
