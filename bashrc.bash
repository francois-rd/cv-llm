#!/bin/bash

# Find the path of where this script file lives.
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

# Add main and plugin code to PYTHONPATH.
export PYTHONPATH="$SCRIPT_DIR"/src:"$SCRIPT_DIR"/plugins

# Environment variables for launching without commands and configs.
export DEFAULT_CONFIG_DIR="$SCRIPT_DIR"/launch
export DEFAULT_COMMAND="test.launch"

# Alias for program entry.
launch () {
  pushd "$DEFAULT_CONFIG_DIR" > /dev/null || exit
  python "$SCRIPT_DIR"/src/main.py "$@"
  popd > /dev/null || exit
}
export -f launch

# Basic terminal auto-complete.
complete -W "
docx.to.json
segment
test.launch
" launch

