#!/bin/bash
# python codesplit.py <repos_info.json> <files_info.json> <chromadb_host> <chromadb_port> <chromadb_collection> <git_repo_name> <dir_with_local_copy_of_git_repo>
#
# ox.python.oxpython
# ox.docs.arch
# ox.org.sw
# ox.queue.backend
# ox.queue.gpu.shared
# ox.runtime.tosa.reference
# ox.base.lib
# intel-extension-for-pytorch
# tt-metal
#
python codesplit.py `pwd`/index/repos_info.json `pwd`/index/files_info.json 127.0.0.1 8000 monorepo_collection oxpython /home/mdharmap/working/for-indexing/oxpython cpu
