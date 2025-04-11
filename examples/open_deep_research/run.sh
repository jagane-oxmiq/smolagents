#!/bin/bash

#python run_ox.py \
#    --chromadb-host 127.0.0.1 \
#    --chromadb-port 8000 \
#    --chromadb-collection oxrag_collection_1 \
#    --local-dir /home/mdharmap/index \
#    --llm-host 127.0.0.1 \
#    --llm-port 11434 \
#    --model QwQ-32B \
#    'How can I intercept torch.distributed calls?'
python run_ox.py \
    --chromadb-host 127.0.0.1 \
    --chromadb-port 8000 \
    --chromadb-collection oxrag_collection \
    --local-dir /home/mdharmap/index1 \
    --llm-host 127.0.0.1 \
    --llm-port 11434 \
    --model QwQ-32B \
    --logs './logs' \
    'How can I intercept torch.distributed calls in oxpython?'
