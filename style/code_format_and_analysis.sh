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

echo "BLACK: %s, PYLINT: %s\n" "$BLACK" "$PYLINT"
# Run BLACK if -b is specified
if [ "$BLACK" = true ] ; then
  echo "[BLACK] Start to check code style and auto format"
  # https://github.com/psf/BLACK/issues/1802
  BLACK --line-length=100 ../
fi

# Run PYLINT if -p is specified
if [ "$PYLINT" = true ] ; then
  echo "[PYLINT] Start code analysis and check,
  we need to manually fix all the warnings mentioned below before commit! "
  export PYTHONPATH=${ROOT_DIR}/hugegraph-llm/src:${ROOT_DIR}/hugegraph-python-client/src
  PYLINT --rcfile=${ROOT_DIR}/style/PYLINT.conf ${ROOT_DIR}/hugegraph-llm
  #PYLINT --rcfile=${ROOT_DIR}/style/PYLINT.conf ${ROOT_DIR}/hugegraph-python-client
fi