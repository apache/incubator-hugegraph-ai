#! /bin/bash

BLACK=false
PYLINT=false
ROOT_DIR=$(pwd)
echo ${ROOT_DIR}
# Parse command line arguments
while getopts ":bp" opt; do
  case ${opt} in
    b )
      BLACK=true
      ;;
    p )
      PYLINT=true
      ;;
    \? )
      echo "Usage: cmd [-b] [-p]"
      ;;
  esac
done

# If no arguments were provided, run both BLACK and PYLINT
if [ "$BLACK" = false ] && [ "$PYLINT" = false ]; then
  BLACK=true
  PYLINT=true
fi

# Run BLACK if -b is specified
if [ "$BLACK" = true ] ; then
  echo "[black] Start to check code style and auto format"
  # https://github.com/psf/BLACK/issues/1802
  black --line-length=120 ${ROOT_DIR}
fi

# Run PYLINT if -p is specified
if [ "$PYLINT" = true ] ; then
  echo "[pylint] Start code analysis and check,
  we need to manually fix all the warnings mentioned below before commit! "
  export PYTHONPATH=${ROOT_DIR}/hugegraph-llm/src:${ROOT_DIR}/hugegraph-python-client/src:${ROOT_DIR}/hugegraph-ml/src
  pylint --rcfile=${ROOT_DIR}/style/pylint.conf ${ROOT_DIR}/hugegraph-llm
  pylint --rcfile=${ROOT_DIR}/style/pylint.conf ${ROOT_DIR}/hugegraph-ml
  pylint --rcfile=${ROOT_DIR}/style/pylint.conf --disable C0103 ${ROOT_DIR}/hugegraph-python-client
fi
