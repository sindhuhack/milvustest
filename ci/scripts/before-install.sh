#!/bin/bash

set -ex

export CCACHE_COMPRESS=1
export CCACHE_COMPRESSLEVEL=5
export CCACHE_COMPILERCHECK=content
export PATH=/usr/lib/ccache/:$PATH
export CORE_BUILD_DIR="$HOME/milvus_build"
export CCACHE_BASEDIR="$HOME/milvus_build"

set +ex
