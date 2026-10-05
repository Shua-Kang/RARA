"""Target + pre-training zarr datasets sampled at fixed per-batch ratios.

Part 0 is the TARGET: its sequences define an epoch (each seen once per epoch), and the
validation split is carved from it only. Other parts are cycled through their own
permutations. With mix_ratios [1, 1] and batch 128 every batch is 64 target + 64
pre-training samples, and one epoch of co-training gives the target exactly the exposure
of one target-only epoch.

Buffers/numba fill path copied from diffusion_policy.dataset.robot_image_dataset so the
per-batch cost is unchanged.
"""
from typing import Dict, List
import atexit
import copy
import json
import os
import shutil
import subprocess
import time
import numpy as np
import torch
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.robot_image_dataset import batch_sample_sequence
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer

LOWDIM_KEYS = ["state", "action"]


def zarr_camera_keys(path):
    """Camera arrays present in a zarr built by hdf5_to_zarr.py (e.g. ['head_camera', 'right_camera'])."""
    root = zarr.open(path, mode="r")
    cams = root.attrs.get("cameras")
    if cams is None:                                     # v1 zarrs carry no attrs: detect by name
        cams = [k for k in root["data"].keys() if k.endswith("_camera")]
    return list(cams)


def obs_key(zarr_key):                                   # head_camera -> head_cam (shape_meta / policy naming)
    return zarr_key.replace("_camera", "_cam")


# ---------------------------------------------------------------------------------------------------------------
# Replay buffers: private copy per process (single-GPU) or ONE copy per node shared by all DDP ranks.
# Under torchrun (WORLD_SIZE > 1) local rank 0 materialises every zarr array once into /dev/shm (tmpfs pages are
# charged to the job's memory cgroup once), the other local ranks wait and map the same files read-only, and every
# rank builds a numpy-backend ReplayBuffer on the mapped arrays. Without this a 4-rank job needs 4 x 461 KB/frame
# (a 330k-frame pool: 4 x 153 GB) and cannot fit a 384 GB node. Disable with RTDP_SHARED_BUFFER=0.
# ---------------------------------------------------------------------------------------------------------------
def _shm_dir():
    tag = os.environ.get("SLURM_JOB_ID") or os.environ.get("TORCHELASTIC_RUN_ID") or f"pid{os.getppid()}"
    base = os.environ.get("RTDP_SHM_BASE", "/dev/shm")
    return os.path.join(base, f"rtdp_{tag}")


def _cleanup_stale(base, keep):
    """Remove /dev/shm/rtdp_* left by earlier jobs of this user on the same node (job no longer in the queue)."""
    try:
        for d in os.listdir(base):
            full = os.path.join(base, d)
            if not d.startswith("rtdp_") or full == keep or not os.path.isdir(full): continue
            if os.stat(full).st_uid != os.getuid(): continue
            jid = d[len("rtdp_"):]
            alive = False
            if jid.isdigit():
                try:
                    alive = subprocess.run(["squeue", "-h", "-j", jid], capture_output=True, text=True, timeout=20).stdout.strip() != ""
                except Exception:
                    alive = time.time() - os.stat(full).st_mtime < 6 * 3600   # cannot ask slurm: keep anything recent
            if not alive:
                shutil.rmtree(full, ignore_errors=True); print(f"[shm] removed stale {full}", flush=True)
    except Exception as e:
        print(f"[shm] stale cleanup skipped: {e}", flush=True)


def _write_shared(zarr_path, keys, out_dir, idx, chunk=2048):
    """Copy data/<key> and meta/episode_ends of one zarr into .npy memmaps (chunked: no second full copy in RAM)."""
    root = zarr.open(zarr_path, mode="r"); spec = {}
    for k in keys:
        src = root["data"][k]; dst_path = os.path.join(out_dir, f"{idx}_{k}.npy")
        dst = np.lib.format.open_memmap(dst_path, mode="w+", dtype=src.dtype, shape=src.shape)
        for s0 in range(0, src.shape[0], chunk):
            dst[s0:s0 + chunk] = src[s0:s0 + chunk]
        dst.flush(); del dst
        spec[k] = dst_path
    ee_path = os.path.join(out_dir, f"{idx}_episode_ends.npy")
    np.save(ee_path, np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64))
    return {"data": spec, "episode_ends": ee_path, "zarr": zarr_path}


def _open_shared(spec):
    data = {k: np.load(path, mmap_mode="r").view(np.ndarray) for k, path in spec["data"].items()}   # plain ndarray views (numba-safe)
    meta = {"episode_ends": np.load(spec["episode_ends"])}
    return ReplayBuffer({"data": data, "meta": meta})


def _barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def load_replay_buffers(zarr_paths, keys):
    world = int(os.environ.get("WORLD_SIZE", "1")); local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world <= 1 or os.environ.get("RTDP_SHARED_BUFFER", "1") == "0":
        return [ReplayBuffer.copy_from_path(p, keys=keys) for p in zarr_paths]
    d = _shm_dir(); manifest = os.path.join(d, "manifest.json")
    if local_rank == 0:
        _cleanup_stale(os.path.dirname(d), d)
        if os.path.isdir(d): shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        t0 = time.time(); specs = [_write_shared(p, keys, d, i) for i, p in enumerate(zarr_paths)]
        tmp = manifest + ".tmp"; json.dump({"zarr_paths": list(zarr_paths), "keys": list(keys), "specs": specs}, open(tmp, "w")); os.replace(tmp, manifest)
        nbytes = sum(os.path.getsize(v) for sp in specs for v in sp["data"].values())
        print(f"[shm] rank0 wrote {len(specs)} buffers ({nbytes/2**30:.1f} GiB) to {d} in {time.time()-t0:.0f}s; shared by {world} ranks", flush=True)
        atexit.register(lambda: shutil.rmtree(d, ignore_errors=True))
    _barrier()
    if not os.path.exists(manifest):          # no process group yet (should not happen under torchrun): poll
        t0 = time.time()
        while not os.path.exists(manifest):
            if time.time() - t0 > 7200: raise RuntimeError(f"[shm] rank {local_rank}: manifest {manifest} never appeared")
            time.sleep(5)
    m = json.load(open(manifest))
    assert m["zarr_paths"] == list(zarr_paths) and m["keys"] == list(keys), (m["zarr_paths"], zarr_paths)
    return [_open_shared(sp) for sp in m["specs"]]


class MixedRobotImageDataset(BaseImageDataset):

    def __init__(self, zarr_paths: List[str], mix_ratios=None, horizon=1, pad_before=0, pad_after=0,
                 seed=42, val_ratio=0.0, batch_size=128):
        super().__init__()
        zarr_paths = list(zarr_paths)
        mix_ratios = [1.0] * len(zarr_paths) if mix_ratios is None else [float(r) for r in mix_ratios]
        assert len(mix_ratios) == len(zarr_paths) and all(r > 0 for r in mix_ratios)
        self.camera_keys = zarr_camera_keys(zarr_paths[0])
        for p in zarr_paths[1:]:
            assert zarr_camera_keys(p) == self.camera_keys, (p, zarr_camera_keys(p), self.camera_keys)
        keys = self.camera_keys + LOWDIM_KEYS
        self.buffers_rb = load_replay_buffers(zarr_paths, keys)
        self.masks, self.samplers = [], []
        for i, rb in enumerate(self.buffers_rb):
            val_mask = get_val_mask(n_episodes=rb.n_episodes, val_ratio=(val_ratio if i == 0 else 0.0), seed=seed)
            self.masks.append(~val_mask)
            self.samplers.append(SequenceSampler(replay_buffer=rb, sequence_length=horizon, pad_before=pad_before,
                                                 pad_after=pad_after, episode_mask=~val_mask))
        self.zarr_paths, self.mix_ratios = zarr_paths, mix_ratios
        self.horizon, self.pad_before, self.pad_after, self.seed = horizon, pad_before, pad_after, seed
        self.batch_size = batch_size
        # per-part batch shares (sum to batch_size, part 0 gets the rounding remainder)
        tot = sum(mix_ratios)
        self.shares = [int(round(batch_size * r / tot)) for r in mix_ratios]
        self.shares[0] += batch_size - sum(self.shares)
        assert all(s > 0 for s in self.shares), self.shares
        self.is_val = False
        self._alloc_buffers(batch_size)

    def _alloc_buffers(self, batch_size):
        seq = self.horizon
        self.buffers = {k: np.zeros((batch_size, seq, *v.shape[1:]), dtype=v.dtype)
                        for k, v in self.buffers_rb[0].items()}
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}

    # ---- epoch structure: one pass over the target's train sequences
    def __len__(self) -> int:
        return len(self.samplers[0])

    @property
    def batches_per_epoch(self) -> int:
        return len(self.samplers[0]) // self.shares[0]

    def get_validation_dataset(self):
        val = copy.copy(self)
        rb = self.buffers_rb[0]
        val.samplers = [SequenceSampler(replay_buffer=rb, sequence_length=self.horizon, pad_before=self.pad_before,
                                        pad_after=self.pad_after, episode_mask=~self.masks[0])]
        val.buffers_rb = [rb]
        val.shares = [self.batch_size]
        val.is_val = True
        val._alloc_buffers(self.batch_size)
        return val

    def get_normalizer(self, mode="limits", **kwargs):
        data = {"action": np.concatenate([rb["action"][:] for rb in self.buffers_rb]),
                "agent_pos": np.concatenate([rb["state"][:] for rb in self.buffers_rb])}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for k in self.camera_keys:
            normalizer[obs_key(k)] = get_image_range_normalizer()
        return normalizer

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """idx: list of per-part index arrays (from MixedBatchSampler)."""
        assert isinstance(idx, (list, tuple)) and len(idx) == len(self.samplers)
        off = 0
        for part, ids in enumerate(idx):
            n = len(ids)
            if n == 0:
                continue
            sampler = self.samplers[part]
            for k, v in sampler.replay_buffer.items():
                batch_sample_sequence(self.buffers[k][off:off + n], v, sampler.indices, np.asarray(ids), self.horizon)
            off += n
        assert off == self.batch_size, (off, self.batch_size)
        return self.buffers_torch

    def postprocess(self, samples, device):
        obs = {obs_key(k): samples[k].to(device, non_blocking=True) / 255.0 for k in self.camera_keys}
        obs["agent_pos"] = samples["state"].to(device, non_blocking=True)
        return {"obs": obs, "action": samples["action"].to(device, non_blocking=True)}


class MixedBatchSampler:
    """Yields [idx_part0, idx_part1, ...] per batch. Part 0 is permuted once per epoch
    (drop_last); the other parts cycle through their own permutations across epochs."""

    def __init__(self, dataset: MixedRobotImageDataset, shuffle=True, seed=0, rank=0, world_size=1):
        """rank/world_size (DDP): every rank draws the SAME per-epoch permutation of the target (part 0)
        and consumes its own contiguous 1/world_size slice, so the ranks together make exactly one pass
        per epoch; the other parts use a rank-specific stream."""
        self.ds, self.shuffle = dataset, shuffle
        self.rank, self.world_size = rank, world_size
        self.rng = np.random.default_rng(seed + 7919 * rank)     # parts >= 1
        self.rng0 = np.random.default_rng(seed)                  # part 0: identical on every rank
        self.sizes = [len(s) for s in dataset.samplers]
        self.shares = dataset.shares
        self.num_batch = (self.sizes[0] // self.shares[0]) // world_size
        self._perm = [None] * len(self.sizes)
        self._pos = [0] * len(self.sizes)

    def _take(self, part, n):
        out = []
        while n > 0:
            if self._perm[part] is None or self._pos[part] >= self.sizes[part]:
                self._perm[part] = self.rng.permutation(self.sizes[part]) if self.shuffle else np.arange(self.sizes[part])
                self._pos[part] = 0
            take = min(n, self.sizes[part] - self._pos[part])
            out.append(self._perm[part][self._pos[part]:self._pos[part] + take])
            self._pos[part] += take
            n -= take
        return np.concatenate(out)

    def __iter__(self):
        # fresh target permutation each epoch (shared across ranks), this rank's contiguous slice
        perm0 = self.rng0.permutation(self.sizes[0]) if self.shuffle else np.arange(self.sizes[0])
        n = self.num_batch * self.shares[0]
        self._perm[0], self._pos[0] = perm0[self.rank * n:(self.rank + 1) * n], 0
        for _ in range(self.num_batch):
            yield [self._take(p, s) for p, s in enumerate(self.shares)]

    def __len__(self):
        return self.num_batch
