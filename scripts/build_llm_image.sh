#!/bin/bash

set -e

tag="v0.0.1"

script_dir=$(realpath "$(dirname "$0")")

cd "${script_dir}/../docker"

name="iregistry.baidu-int.com/hugegraph-vermeer/hugegraph-llm:${tag}"

docker build -f Dockerfile.llm -t ${name} ..