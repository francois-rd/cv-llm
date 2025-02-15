import sys
import os

import coma

from .base import Configs as Cfgs, init
from ..io import PathConfig, load_docx, save_json


def docx_to_json(path: PathConfig):
    for root, _, files in os.walk(path.docx_transcript_dir):
        for filename in files:
            # Extract the 'assign ID' to create the output filename.
            aid = filename.split("_")[0]
            output_file = str(os.path.join(path.json_transcript_dir, aid + ".json"))

            # Convert the data and write to file in JSON format.
            lines = load_docx(str(os.path.join(root, filename)))
            save_json(output_file, lines, indent=4)


def register():
    """Registers all known commands with Coma."""
    coma.register("test.launch", lambda: print("Successfully launched."))
    coma.register("docx.to.json", docx_to_json, **Cfgs.add(Cfgs.paths))


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
