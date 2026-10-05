"""RARA Stage 1: encoder-only alignment of target frames to geometry-matched frames of the retrieved data.

  * descriptor of a frame = (end-effector offset to the grasp point, frames-to-grasp / episode length, gripper);
  * positives = the m+ retrieved frames closest in descriptor space; negatives = random retrieved frames farther
    than the median distance (closer ones are left out of the denominator);
  * multi-positive InfoNCE (cosine, temperature tau) between the trainable target features and the frozen
    pre-trained features of the retrieved frames, plus an anchor that keeps the encoder output on retrieved frames
    at its pre-trained value;
  * only the two camera encoders and a small projector are trained; the projector is discarded and the encoder is
    saved as a light checkpoint that Stage 2 (MODE=rara) loads as its initialisation.

usage: stage1.py --init <pretrain.pt> --target <target.zarr> --target_geom <target.npz>
                 --pool <retrieved.zarr> --pool_geom <retrieved.npz> --out <dir>
                 [--steps 3000 --lr 1e-5 --lam 1.0 --mpos 4 --tau 0.07]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch, zarr
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_zarr(path):
    z = zarr.open(path, "r")
    ends = z["meta/episode_ends"][:]
    return z, ends


class Projector(nn.Module):
    def __init__(self, d_in, d_out=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Linear(512, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def encode_frames(policy, z, idx, device, train=True):
    """Concatenated per-camera encoder features (B, 1024) for zarr frame indices idx."""
    enc = policy.obs_encoder
    feats = []
    for cam in ("head_camera", "right_camera"):
        key = cam.replace("camera", "cam")
        x = torch.from_numpy(z[f"data/{cam}"].get_orthogonal_selection((np.sort(idx),))[np.argsort(np.argsort(idx))]).to(device).float() / 255.0
        m, tf = enc.key_model_map[key], enc.key_transform_map[key]
        feats.append(m(tf(x)))
    return torch.cat(feats, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--target", required=True, help="target zarr")
    ap.add_argument("--target_geom", required=True, help="target geometry npz (desc, lens)")
    ap.add_argument("--pool", required=True, help="retrieved pre-training zarr")
    ap.add_argument("--pool_geom", required=True, help="retrieved pre-training geometry npz (desc, lens)")
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the anchor term on retrieved frames")
    ap.add_argument("--mpos", type=int, default=4); ap.add_argument("--neg_q", type=float, default=0.5,
                    help="retrieved frames above this quantile of descriptor distance are eligible negatives")
    ap.add_argument("--tau", type=float, default=0.07); ap.add_argument("--bt", type=int, default=32, help="target frames per step")
    ap.add_argument("--bneg", type=int, default=64, help="random negatives per step")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", default="", help="wandb project (empty = no wandb)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    device = torch.device(a.device)
    from workspace import load_policy, save_light_checkpoint
    from omegaconf import OmegaConf
    import copy
    policy, cfg, payload = load_policy(a.init, device)
    ref = copy.deepcopy(policy.obs_encoder).eval().requires_grad_(False)
    policy.obs_encoder.train().requires_grad_(True)

    zt, et = load_zarr(a.target); zp, ep_ = load_zarr(a.pool)
    def geom(npz):
        g = np.load(npz); return g["desc"], g["lens"]
    dt, lt = geom(a.target_geom); dp, lp = geom(a.pool_geom)
    assert np.array_equal(np.cumsum(lt), et), "target zarr / geometry episode lengths differ"
    assert np.array_equal(np.cumsum(lp), ep_), "retrieved zarr / geometry episode lengths differ"
    # standardise descriptors with pool statistics; distances in that space define positives
    mu, sd = dp.mean(0), dp.std(0) + 1e-6
    Dt, Dp = (dt - mu) / sd, (dp - mu) / sd
    # nearest pool frames for every target frame (chunked brute force)
    pos = np.zeros((len(Dt), a.mpos), dtype=np.int64); far_thr = np.zeros(len(Dt), dtype=np.float32)
    Dp_t = torch.from_numpy(Dp).to(device)
    for i in range(0, len(Dt), 512):
        d = torch.cdist(torch.from_numpy(Dt[i:i + 512]).to(device), Dp_t)
        pos[i:i + 512] = torch.topk(d, a.mpos, largest=False).indices.cpu().numpy()
        far_thr[i:i + 512] = torch.quantile(d, a.neg_q, dim=1).cpu().numpy()
    print(f"[stage1] target frames {len(Dt)} retrieved frames {len(Dp)} mpos {a.mpos}; mean pos dist "
          f"{np.mean([np.linalg.norm(Dt[i]-Dp[pos[i,0]]) for i in range(0,len(Dt),50)]):.3f}", flush=True)

    with torch.no_grad():   # frozen pool features (pre-trained), computed once
        pool_ref = torch.cat([encode_frames(policy, zp, np.arange(i, min(i + 256, len(Dp))), device)
                              for i in range(0, len(Dp), 256)]).cpu()
    proj = Projector(pool_ref.shape[1]).to(device)
    groups = [{"params": policy.obs_encoder.parameters(), "lr": a.lr}, {"params": proj.parameters(), "lr": 1e-4}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / a.steps))))
    run = None
    if a.wandb:
        import wandb
        run = wandb.init(project=a.wandb, entity=os.environ.get("WANDB_ENTITY"), name=os.path.basename(a.out.rstrip("/")), config=vars(a))
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "stage1_log.jsonl"), "a")
    t0 = time.time()
    for step in range(a.steps):
        ti = rng.choice(len(Dt), a.bt, replace=False)
        pi = pos[ti]                                            # (bt, mpos) positives
        # negatives: random pool frames that are far from EVERY target in the batch's own row (masked below)
        ni = rng.choice(len(Dp), a.bneg, replace=False)
        cand = np.unique(np.concatenate([pi.reshape(-1), ni]))
        zt_feat = proj(encode_frames(policy, zt, ti, device))               # trainable target side
        with torch.no_grad():
            zc_ref = proj(pool_ref[cand].to(device))                       # frozen pre-trained pool side (projector shared)
        sim = zt_feat @ zc_ref.T / a.tau                                    # (bt, |cand|)
        col = {c: j for j, c in enumerate(cand)}
        pos_mask = torch.zeros_like(sim, dtype=torch.bool)
        for r in range(a.bt):
            for c in pi[r]: pos_mask[r, col[c]] = True
        # mask out pool frames that are geometrically close to the target (ambiguous, neither pos nor neg)
        dist = torch.cdist(torch.from_numpy(Dt[ti]).to(device), Dp_t[cand])
        amb = (dist < torch.from_numpy(far_thr[ti]).to(device)[:, None]) & ~pos_mask
        sim = sim.masked_fill(amb, float("-inf"))
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        l_con = -(torch.logsumexp(logp.masked_fill(~pos_mask, float("-inf")), dim=1)).mean()
        # anchor: pool frames through the trainable encoder must stay at their pre-trained features
        ai = rng.choice(len(Dp), a.bt, replace=False)
        l_keep = F.mse_loss(encode_frames(policy, zp, ai, device), pool_ref[ai].to(device))
        loss = l_con + a.lam * l_keep
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.obs_encoder.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 20 == 0:
            with torch.no_grad():
                top1 = (sim.argmax(1)[:, None] == torch.as_tensor([[col[c] for c in row] for row in pi], device=device)).any(1).float().mean().item()
            d = dict(step=step, loss=loss.item(), l_con=l_con.item(), l_keep=l_keep.item(), top1=top1,
                     lr=sched.get_last_lr()[0], elapsed_min=(time.time() - t0) / 60)
            log.write(json.dumps(d) + "\n"); log.flush()
            if run: run.log(d, step=step)
            if step % 200 == 0: print("[stage1]", d, flush=True)
    policy.eval()
    out_ckpt = os.path.join(a.out, "checkpoints", "aligned.pt")
    save_light_checkpoint(out_ckpt, cfg, policy, payload.get("epoch", 0), payload.get("global_step", 0))
    json.dump(vars(a), open(os.path.join(a.out, "stage1_args.json"), "w"), indent=1)
    if run: run.finish()
    print("[stage1] saved", out_ckpt, flush=True)


if __name__ == "__main__":
    main()
