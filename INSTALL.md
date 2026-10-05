# Installation

Two Python environments: one for training and the policy server (`requirements.txt`, CUDA 12.1 PyTorch), one for the
RoboTwin 2.0 simulator (only needed for evaluation). They talk over a local socket.

```
<workdir>/
  RARA/        this repository  (.venv = training environment)
  RoboTwin/    RoboTwin 2.0     (.venv = simulator environment)
```

## 1. Training environment

```bash
cd RARA
uv venv .venv --python 3.10
uv pip install -p .venv -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

## 2. RoboTwin 2.0

The training scripts import the Diffusion Policy code from RoboTwin's `XPolicyLab/policy/DP`, so clone RoboTwin
and its `XPolicyLab` submodule even if you only train. The simulator install below is needed for evaluation.

```bash
cd ..
git clone https://github.com/Shua-Kang/my_roboTwin2.0.git RoboTwin && cd RoboTwin
git checkout aafd43ce76ad6240329d982ec5d1c5117385fe70     # pinned RoboTwin commit (the patch below is made for it)
git submodule update --init XPolicyLab && (cd XPolicyLab && git checkout 482367a)   # Diffusion Policy code
git apply ../RARA/robotwin/workspace_override.patch       # common object workspace for the pre-training data
cp ../RARA/robotwin/task_config/*.yml env_cfg/task_config/  # eval_id / eval_ood settings for the pill task
```

Install the simulator following RoboTwin's instructions. With uv (CUDA 12.4 toolkit and gcc >= 9 on `PATH`):

```bash
uv venv .venv --python 3.10 && PY=$PWD/.venv/bin/python
uv pip install -p $PY -r scripts/requirements.txt
uv pip install -p $PY torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124
uv pip install -p $PY "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
uv pip install -p $PY -e XPolicyLab
S=$($PY -c "import sapien,os;print(os.path.dirname(sapien.__file__))"); sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $S/wrapper/urdf_loader.py
M=$($PY -c "import mplib,os;print(os.path.dirname(mplib.__file__))");  sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $M/planner.py
(cd envs && git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git && cd curobo && uv pip install -p $PY -e . --no-build-isolation)
uv pip install -p $PY warp-lang==1.12.0 setuptools==69.5.1
(cd assets && $PY _download.py && for z in background_texture embodiments objects; do unzip -q -o $z.zip && rm $z.zip; done)
$PY scripts/update_embodiment_config_path.py
```

Evaluation needs an NVIDIA GPU of the Ampere generation or newer (curobo) and an `ffmpeg` with libx264 for rollout
videos. Training runs on any recent CUDA GPU.

If your environments live elsewhere, set `DP_PY`, `RT_PY` and `RT` before running the scripts (see `scripts/env.sh`).
