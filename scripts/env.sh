# Source before any script. Override any variable beforehand if your layout differs.
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export RT=${RT:-$REPO/../RoboTwin}                       # RoboTwin 2.0 checkout (see INSTALL.md)
export DP_PY=${DP_PY:-$REPO/.venv/bin/python}            # training / policy-server environment
export RT_PY=${RT_PY:-$RT/.venv/bin/python}              # RoboTwin simulator environment (evaluation only)
export DATA=${DATA:-$REPO/data}                          # zarrs, geometry files, weights (scripts/download.sh)
export RUNS=${RUNS:-$REPO/runs}                          # training outputs
export EVAL=${EVAL:-$REPO/eval_out}                      # evaluation outputs
export PYTHONPATH=$REPO/rara:$RT/XPolicyLab/policy/DP${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTHONWARNINGS=ignore
mkdir -p "$DATA" "$RUNS" "$EVAL"
