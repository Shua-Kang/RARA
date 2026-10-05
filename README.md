# RARA: Retrieval-Anchored Representation Alignment

RARA adapts a pre-trained robot policy to a new task by retrieving the related part of its pre-training data and
keeping the policy's visual representation aligned to it during fine-tuning.

<p align="center"><img src="assets/teaser.png" width="90%"></p>

## Quick start

**1. Install** the training environment and RoboTwin 2.0 (details in [INSTALL.md](INSTALL.md)):

```bash
git clone https://github.com/Shua-Kang/RARA.git && cd RARA
uv venv .venv --python 3.10
uv pip install -p .venv -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
# then RoboTwin 2.0 next to this folder: see INSTALL.md
```

**2. Download** the pill demonstrations, the pre-training data and the checkpoints:

```bash
bash scripts/download.sh            # add "small" to skip the 77 GB pre-training data (steps 3 and 4 then use the released weights and selection)
```

**3. Pre-train** (optional; the pre-trained policy is part of the download):

```bash
MODE=pretrain bash scripts/train.sh      # 2000 demonstrations, 60 epochs, batch 256; 4 GPUs, ~160 GB host RAM
```

To use your own pre-trained policy in the steps below, set `P=runs/pretrain_s42/checkpoints/epoch0060.pt`.

**4. Select** (optional; the 250 pre-selected episodes are part of the download) the pre-training episodes closest to the pill demonstrations:

```bash
bash scripts/select.sh                   # pick-and-place episodes ranked by trajectory DTW to the target
```

**5. Train** FT and RARA from the pre-trained policy:

```bash
MODE=ft   bash scripts/train.sh          # FT
bash scripts/stage1.sh                   # RARA, stage 1: align the encoder to the selected data
MODE=rara bash scripts/train.sh          # RARA, stage 2: fine-tune with the anchor to the pre-trained policy
# MODE=coft bash scripts/train.sh        # Co-FT (optional baseline)
```

**6. Evaluate** (last 3 checkpoints x 2 seed blocks x 25 episodes; use `runs/pill_ft_s42` for FT):

```bash
bash scripts/evaluate.sh runs/pill_rara_s42 id     # demonstration configuration
bash scripts/evaluate.sh runs/pill_rara_s42 ood    # wider object positions
```

The released checkpoints can be evaluated directly, e.g.
`CKPT=data/weights/rara.pt SPLIT=id SEEDSTART=100000 TAG=rara_id bash scripts/eval.sh`.

## Expected results

ID success rate (%) on move pill bottle to pad (`scripts/evaluate.sh <run> id`: last three checkpoints
x 2 seed blocks x 25 episodes = 150 rollouts per method):

| | FT | Co-FT | RARA |
|---|---|---|---|
| ID | 44.0 | 44.7 | **59.3** |

## Data and checkpoints

| | Hugging Face |
|---|---|
| Pill demonstrations (50) + retrieved pre-training episodes (250) | [`ShuaKang/robotwin_pill_rara`](https://huggingface.co/datasets/ShuaKang/robotwin_pill_rara) |
| Pre-training set (2000 demonstrations, 5 tasks) | [`ShuaKang/my_roboTwin2.0_training`](https://huggingface.co/datasets/ShuaKang/my_roboTwin2.0_training) (`dp_zarr_ws/`) |
| Pre-trained policy | [`ShuaKang/RARA-robotwin-pretrain`](https://huggingface.co/ShuaKang/RARA-robotwin-pretrain) |
| Fine-tuned policies (FT, Co-FT, RARA) | [`ShuaKang/RARA-robotwin-pill`](https://huggingface.co/ShuaKang/RARA-robotwin-pill) |

The retrieved set keeps the pick-and-place episodes of the pre-training data (task filter) and, among them, the 250
whose grasp-relative end-effector trajectories are closest to the pill demonstrations under dynamic time warping.

## Acknowledgements

Built on [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) and
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy).
