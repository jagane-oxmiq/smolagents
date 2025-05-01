#!/bin/bash
#
#    --llm-host 192.168.2.114 \
#    --llm-port 11434 \

python listener.py \
    --directory ./input_json \
    --destination ./processing_finished_json \
    --chromadb-host 127.0.0.1 \
    --chromadb-port 8000 \
    --chromadb-collection monorepo_collection \
    --local-dir ./ox-indexer/index \
    --llm-host 127.0.0.1 \
    --llm-port 8080 \
    --model QwQ-32B \
    --logs './static/logs'
