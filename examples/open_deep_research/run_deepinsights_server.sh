#!/bin/bash
#
if [ "$1" == "t4" ] ; then
    HH="10.11.2.23"
    PP="11434"
elif [ "$1" == "r1" ] ; then
    HH="127.0.0.1"
    PP="8080"
else
    echo "Usage $0 t4|r1"
    exit 255
fi

python listener.py \
    --directory ./input_json \
    --destination ./processing_finished_json \
    --chromadb-host 127.0.0.1 \
    --chromadb-port 8000 \
    --chromadb-collection oxpython_collection \
    --local-dir ./ox-indexer/index \
    --llm-host $HH \
    --llm-port $PP \
    --model QwQ-32B \
    --logs './static/logs'
