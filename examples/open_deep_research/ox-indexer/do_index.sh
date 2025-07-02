#!/bin/bash

#EMBEDDING_MODEL='/home/mdharmap/models/gte-Qwen2-1.5B-instruct-half'
EMBEDDING_MODEL='nomic-ai/nomic-embed-code'
EMBEDDING_MODEL_URL='http://127.0.0.1:8001/'
DEVICE='cuda:7'
#DEVICE='cpu'

python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/${USER}/working/oxpython/components/cpython/examples dontsummarize "${EMBEDDING_MODEL}" "${EMBEDDING_MODEL_URL}" "${DEVICE}"
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/${USER}/working/oxpython/components/cpython/ox-backend dontsummarize "${EMBEDDING_MODEL}" "${EMBEDDING_MODEL_URL}" "${DEVICE}"
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/${USER}/working/oxpython/docs/chapters dontsummarize "${EMBEDDING_MODEL}" "${EMBEDDING_MODEL_URL}" "${DEVICE}"
#
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/${USER}/working/oxpython/components/backends/oxsim dontsummarize "${EMBEDDING_MODEL}" "${EMBEDDING_MODEL_URL}" "${DEVICE}"
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/${USER}/working/oxpython/components/backends/protosim dontsummarize "${EMBEDDING_MODEL}" "${EMBEDDING_MODEL_URL}" "${DEVICE}"
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/${USER}/working/oxpython/components/backends/intel dontsummarize "${EMBEDDING_MODEL}" "${EMBEDDING_MODEL_URL}" "${DEVICE}"
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection oxpython /home/${USER}/working/oxpython/components/backends/tenstorrent/ox-src dontsummarize "${EMBEDDING_MODEL}" "${EMBEDDING_MODEL_URL}" "${DEVICE}"
#
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 oxpython_collection micc /home/${USER}/working/terraform-micc dontsummarize "${EMBEDDING_MODEL}" "${EMBEDDING_MODEL_URL}" "${DEVICE}"
