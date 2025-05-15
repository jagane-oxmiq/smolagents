#!/bin/bash

python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/mdharmap/working/for-indexing/oxpython http://127.0.0.1:11434/v1/completions '/home/mdharmap/models/gte-Qwen2-1.5B-instruct-half' cuda:0
