"""Diffusion-policy server for RoboTwin closed-loop evaluation (talks to eval/robotwin_client.py over a local socket):

  {"cmd": "ping"}                                   -> {"ok": True}
  {"cmd": "reset"}                                  -> {"ok": True}   clear obs history (episode start)
  {"cmd": "obs", "joint": (14,), "rgb": {cam: HxWx3}} -> {"ok": True}   push one obs (after each executed action)
  {"joint": (14,), "rgb": {...}, "lang": str, ...}  -> {"chunk": (n_action_steps, 14) f32}

History handling mirrors XPolicyLab/policy/DP dp_runner.DPRunner: keep the last n_obs_steps
observations, pad the front by repeating the oldest one, predict, return the first
n_action_steps joint-space targets (take_action(action_type='qpos') layout).
--random emits uniform noise in the target zarr's per-dim action [min,max] (SR floor).
"""
from __future__ import annotations
import argparse, pickle, socket, struct, sys
from collections import deque
from pathlib import Path
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
SIM2ZARR = {"head": "head_cam", "left_wrist": "left_cam", "right_wrist": "right_cam", "third_view": "third_cam"}


def recv_msg(conn):
    hdr = conn.recv(8, socket.MSG_WAITALL)
    if len(hdr) < 8:
        return None
    (n,) = struct.unpack("<Q", hdr)
    buf = b""
    while len(buf) < n:
        part = conn.recv(min(1 << 20, n - len(buf)))
        if not part:
            return None
        buf += part
    return pickle.loads(buf)


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("<Q", len(data)) + data)


class DPPredictor:
    def __init__(self, ckpt, device="cuda:0"):
        import torch
        from workspace import load_policy
        self.torch = torch
        self.policy, self.cfg, payload = load_policy(ckpt, device)
        self.device = device
        self.n_obs = int(self.cfg.n_obs_steps)
        self.n_act = int(self.cfg.n_action_steps)
        self.obs_keys = [k for k, v in self.cfg.shape_meta.obs.items() if v.type == "rgb"]
        self.hist = deque(maxlen=self.n_obs)
        print(f"[dp_server] loaded {ckpt} epoch={payload.get('epoch')} step={payload.get('global_step')} "
              f"n_obs={self.n_obs} n_act={self.n_act} cams={self.obs_keys}", flush=True)

    def reset(self):
        self.hist.clear()

    def encode(self, joint, rgb):
        frames = rgb if isinstance(rgb, dict) else {"head": rgb}
        o = {}
        for sim, key in SIM2ZARR.items():
            if key in self.obs_keys:
                img = np.asarray(frames[sim], np.uint8)
                assert img.shape == (240, 320, 3), img.shape
                o[key] = np.moveaxis(img, -1, 0).astype(np.float32) / 255.0     # 3,H,W
        o["agent_pos"] = np.asarray(joint, np.float32)
        return o

    def push(self, joint, rgb):
        self.hist.append(self.encode(joint, rgb))

    def predict(self, joint, rgb):
        self.push(joint, rgb)
        hist = list(self.hist)
        hist = [hist[0]] * (self.n_obs - len(hist)) + hist          # front-pad like stack_last_n_obs
        obs = {k: self.torch.from_numpy(np.stack([h[k] for h in hist])[None]).to(self.device)
               for k in hist[0]}
        with self.torch.no_grad():
            out = self.policy.predict_action(obs)
        return out["action"][0, :self.n_act].detach().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--port", type=int, default=15151)
    ap.add_argument("--random-zarr", default=None, help="serve uniform noise in this zarr's action range")
    ap.add_argument("--n-act", type=int, default=6)
    args = ap.parse_args()
    if args.random_zarr:
        act = zarr.open(args.random_zarr, mode="r")["data/action"][:]
        lo, hi = act.min(0), act.max(0)
        rng = np.random.default_rng(0)
        class Rand:
            def reset(self): pass
            def push(self, joint, rgb): pass
            def predict(self, joint, rgb): return rng.uniform(lo, hi, size=(args.n_act, len(lo))).astype(np.float32)
        pred = Rand()
        print(f"[dp_server] RANDOM mode from {args.random_zarr}", flush=True)
    else:
        assert args.ckpt, "--ckpt or --random-zarr required"
        pred = DPPredictor(args.ckpt)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port)); srv.listen(1)
    print(f"[dp_server] listening on 127.0.0.1:{args.port}", flush=True)
    while True:
        conn, _ = srv.accept()
        print("[dp_server] client connected", flush=True)
        while True:
            msg = recv_msg(conn)
            if msg is None:
                break
            cmd = msg.get("cmd")
            if cmd == "ping":
                send_msg(conn, {"ok": True})
            elif cmd == "reset":
                pred.reset(); send_msg(conn, {"ok": True})
            elif cmd == "obs":
                pred.push(msg["joint"], msg["rgb"]); send_msg(conn, {"ok": True})
            else:
                send_msg(conn, {"chunk": pred.predict(msg["joint"], msg["rgb"])})
        conn.close()
        print("[dp_server] client disconnected", flush=True)


if __name__ == "__main__":
    main()
