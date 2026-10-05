#!/usr/bin/env python
"""Select the pre-training episodes RARA anchors to.

1. Task filter: keep the pre-training episodes of the target's skill (pick-and-place for the pill task).
2. Rank them by the mean DTW distance between their grasp-relative end-effector trajectory (5-d per-frame descriptor,
   standardised with pre-training statistics, resampled to 32 points) and the target demonstrations; keep the top K.
3. Write the selected episodes as a zarr (+ geometry npz and a json with the selection).

usage: select.py --pool pretrain.zarr --pool_geom pretrain.npz --pool_episodes pretrain.episodes.json
                 --target_geom pill_target.npz --task place_a2b_right --k 250 --out pill_retrieved
"""
import argparse, collections, json
import numpy as np, zarr
from numcodecs import Blosc


def per_episode(desc, lens, n=32):
    out, s = [], 0
    for L in lens:
        d = desc[s:s + L]; s += L
        out.append(d[np.linspace(0, L - 1, n).astype(int)])
    return np.stack(out)


def dtw_to_set(a, T):
    n = a.shape[0]; J, m = T.shape[0], T.shape[1]
    c = np.linalg.norm(a[None, :, None, :] - T[:, None, :, :], axis=-1)
    D = np.full((J, n + 1, m + 1), np.inf); D[:, 0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[:, i, j] = c[:, i - 1, j - 1] + np.minimum(np.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1])
    return D[:, n, m] / (n + m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True); ap.add_argument("--pool_geom", required=True); ap.add_argument("--pool_episodes", required=True)
    ap.add_argument("--target_geom", required=True); ap.add_argument("--task", default="place_a2b_right")
    ap.add_argument("--k", type=int, default=250); ap.add_argument("--out", required=True, help="output prefix (writes <out>.zarr, <out>.npz, <out>.json)")
    a = ap.parse_args()
    zt, zp = np.load(a.target_geom), np.load(a.pool_geom)
    tasks = [r["task"] for r in json.load(open(a.pool_episodes))]; assert len(tasks) == len(zp["lens"])
    mu, sd = zp["desc"].mean(0), zp["desc"].std(0) + 1e-6
    T = per_episode((zt["desc"] - mu) / sd, zt["lens"]); P = per_episode((zp["desc"] - mu) / sd, zp["lens"])
    cand = [i for i, t in enumerate(tasks) if t == a.task]
    score = {i: float(dtw_to_set(P[i], T).mean()) for i in cand}
    pick = sorted(sorted(cand, key=score.get)[:a.k])
    print(f"task filter '{a.task}': {len(cand)} of {len(tasks)} episodes; keeping the {len(pick)} nearest "
          f"(DTW cutoff {max(score[i] for i in pick):.3f})")
    # zarr of the selected episodes (pool order)
    zi = zarr.open(a.pool, "r"); ends = zi["meta/episode_ends"][:]; starts = np.concatenate([[0], ends[:-1]])
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE); zo = zarr.open(a.out + ".zarr", "w"); arrays, new_ends, total = {}, [], 0
    for i in pick:
        lo, hi = int(starts[i]), int(ends[i])
        for key in zi["data"].keys():
            x = zi["data"][key][lo:hi]
            if key not in arrays:
                arrays[key] = zo.create_dataset(f"data/{key}", shape=(0,) + x.shape[1:], dtype=x.dtype, chunks=(100,) + x.shape[1:], compressor=comp)
            arrays[key].append(x)
        total += hi - lo; new_ends.append(total)
    zo.create_dataset("meta/episode_ends", data=np.array(new_ends, dtype=np.int64))
    # geometry of the selected episodes (for stage 1) and the selection itself
    gs = np.concatenate([[0], np.cumsum(zp["lens"])])
    np.savez(a.out + ".npz", desc=np.concatenate([zp["desc"][gs[i]:gs[i + 1]] for i in pick]), lens=zp["lens"][pick])
    json.dump(dict(task=a.task, k=a.k, episodes=[dict(pool_episode=i, distance=score[i]) for i in pick],
                   composition=dict(collections.Counter(tasks[i] for i in pick))), open(a.out + ".json", "w"), indent=1)
    print(f"wrote {a.out}.zarr ({len(pick)} episodes, {total} frames), {a.out}.npz, {a.out}.json")


if __name__ == "__main__":
    main()
