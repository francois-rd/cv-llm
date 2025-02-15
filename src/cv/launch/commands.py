import sys
import os

import coma

from .base import init


def register():
    """Registers all known commands with Coma."""
    coma.register("test.launch", lambda: print("Successfully launched."))


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
