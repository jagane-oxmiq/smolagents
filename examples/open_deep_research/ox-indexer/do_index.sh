#!/bin/bash

python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/mdharmap/working/oxpython/components/cpython/examples dontsummarize '/home/mdharmap/models/gte-Qwen2-1.5B-instruct-half' cuda:0
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/mdharmap/working/oxpython/components/cpython/ox-backend dontsummarize '/home/mdharmap/models/gte-Qwen2-1.5B-instruct-half' cuda:0
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/mdharmap/working/oxpython/docs/chapters dontsummarize '/home/mdharmap/models/gte-Qwen2-1.5B-instruct-half' cuda:0
#
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/mdharmap/working/oxpython/components/backends/oxsim dontsummarize '/home/mdharmap/models/gte-Qwen2-1.5B-instruct-half' cuda:0
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/mdharmap/working/oxpython/components/backends/protosim dontsummarize '/home/mdharmap/models/gte-Qwen2-1.5B-instruct-half' cuda:0
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/mdharmap/working/oxpython/components/backends/intel dontsummarize '/home/mdharmap/models/gte-Qwen2-1.5B-instruct-half' cuda:0
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/mdharmap/working/oxpython/components/backends/tenstorrent/ox-src dontsummarize '/home/mdharmap/models/gte-Qwen2-1.5B-instruct-half' cuda:0
#
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection micc /home/mdharmap/working/terraform-micc dontsummarize '/home/mdharmap/models/gte-Qwen2-1.5B-instruct-half' cuda:0
