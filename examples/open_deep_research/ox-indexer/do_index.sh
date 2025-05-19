#!/bin/bash

python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/mdharmap/working/oxpython/components/cpython/examples dontsummarize '/home/mdharmap/models/gte-Qwen2-1.5B-instruct-half' cpu
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/mdharmap/working/oxpython/components/cpython/ox-backend dontsummarize '/home/mdharmap/models/gte-Qwen2-1.5B-instruct-half' cpu
#python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/mdharmap/working/oxpython/components/backends dontsummarize '/home/mdharmap/models/gte-Qwen2-1.5B-instruct-half' cuda:0
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/mdharmap/working/oxpython/docs/chapters dontsummarize '/home/mdharmap/models/gte-Qwen2-1.5B-instruct-half' cpu
