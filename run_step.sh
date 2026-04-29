#!/bin/bash
# run_step.sh STEP SCRIPT [ARGS...]
# Runs SCRIPT inside apptainer, updates pipeline state before/after.
set -uo pipefail

STEP="$1";  shift
SCRIPT="$1"; shift
EXTRA="${*:-}"

ROOT="/home/compiling-ganesh/24m0797/workspace/indic-alignment"
PY="/home/compiling-ganesh/24m0797/envs/vllm/bin/python"
APPTAINER="apptainer exec --nv $HOME/images/cuda-custom-amal_latest.sif"

$PY "$ROOT/pipeline_state.py" set "$STEP" running

set +e
$APPTAINER bash -c "
    cd $ROOT
    HTTP_PROXY=http://127.0.0.1:3128 HTTPS_PROXY=http://127.0.0.1:3128
    $PY $ROOT/$SCRIPT $EXTRA
" 2>&1
EXIT=$?
set -e

if [ $EXIT -eq 0 ]; then
    $PY "$ROOT/pipeline_state.py" set "$STEP" done
else
    $PY "$ROOT/pipeline_state.py" set "$STEP" failed "exit=$EXIT"
    exit $EXIT
fi
