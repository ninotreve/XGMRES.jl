#!/bin/sh

SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
FILEPATH="$SCRIPTPATH/$1"

cd $SCRIPTPATH
cd $(git rev-parse --show-toplevel)

if ! (test -f "$FILEPATH"); then
  echo "ERROR: The provided Julia script does not exist. Possible scripts are listed below."
  cd $SCRIPTPATH
  ls *.jl
  exit 1
fi

if ! [ -d "./scripts/logs" ]; then
  mkdir scripts/logs/
fi

export JULIA_NUM_THREADS=1
nohup julia --project=. scripts/$1 > scripts/logs/"${1%.*}"_"$(date +%s)".log &
