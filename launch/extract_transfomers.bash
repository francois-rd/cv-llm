#!/bin/bash

LLM="$1"
shift
if [ -z "$LLM" ]
then
  echo "Missing LLM nickname"
  exit 1
fi
launch extract \
  llm="$LLM" \
  implementation=HF_TRANSFORMERS \
  --transformers-path transformers_cfgs/"$LLM"/transformers.yaml \
  "$@"
