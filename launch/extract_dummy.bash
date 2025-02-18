#!/bin/bash

LLM="$1"
shift
if [ -z "$LLM" ]
then
  LLM=dummy
fi
launch extract llm="$LLM" implementation=DUMMY "$@"
