#!/bin/bash

LLM="$1"
shift
if [ -z "$LLM" ]
then
  LLM=dummy
fi
launch extract \
  llm="$LLM" \
  implementation=DUMMY \
  --transformers-path transformers_cfgs/dummy_transformers.yaml \
  "$@"
