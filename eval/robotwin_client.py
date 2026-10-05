"""Closed-loop RoboTwin 2.0 evaluation client (runs in the RoboTwin environment, cwd = RoboTwin repo root).

Follows RoboTwin's evaluation protocol: for each environment seed the curobo expert is run first and seeds it cannot
solve are skipped; the policy (served by eval/server.py) is then rolled out on the same seed until success or the
task's step limit. The policy predicts 14-d joint-position chunks (--variant joint) and is re-queried every
--replan steps; --history streams every executed observation to the server (n_obs_steps history).

  python robotwin_client.py --task move_pillbottle_pad --variant joint --history --replan 6 \
      --config eval_id --n 25 --port 15151 --cameras head,right_wrist --out <output dir>
"""
from __future__ import annotations
import argparse, json, os, pickle, socket, struct, subprocess, sys
from pathlib import Path
import numpy as np

sys.path.append("./")
from scipy.spatial.transform import Rotation as R

from envs import CONFIGS_PATH  # noqa: E402
import yaml  # noqa: E402
import importlib  # noqa: E402


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("<Q", len(data)) + data)


def recv_msg(conn):
    hdr = conn.recv(8, socket.MSG_WAITALL)
    (n,) = struct.unpack("<Q", hdr)
    buf = b""
    while len(buf) < n:
        buf += conn.recv(min(1 << 20, n - len(buf)))
    return pickle.loads(buf)


def rot6d_to_quat_wxyz(r6):
    a1, a2 = np.asarray(r6[:3], np.float64), np.asarray(r6[3:6], np.float64)
    b1 = a1 / np.linalg.norm(a1)
    a2 = a2 - (b1 @ a2) * b1
    b2 = a2 / np.linalg.norm(a2)
    b3 = np.cross(b1, b2)
    m = np.stack([b1, b2, b3], axis=1)
    q = R.from_matrix(m).as_quat()          # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def pose_to_state_parts(endpose_7, grip):
    """RoboTwin pose [xyz + wxyz quat] + gripper val -> pos3, rot6d6, grip1."""
    p = np.asarray(endpose_7[:3], np.float32)
    q = np.asarray(endpose_7[3:7], np.float64)
    m = R.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    r6 = m[:, :2].T.reshape(6).astype(np.float32)
    return p, r6, np.array([grip], np.float32)


# lerobot camera name -> the key the sim's obs dict uses (see pkl2hdf5.CAMERA_MAP)
SIM_CAM = {"head": "head_camera", "left_wrist": "left_camera",
           "right_wrist": "right_camera", "third_view": "front_camera"}


POLICY_RES = None   # (w, h): resize camera frames to the policy's training resolution before sending (set from --policy-res)


def _to_policy_res(img):
    if POLICY_RES is None or (img.shape[1], img.shape[0]) == tuple(POLICY_RES):
        return img
    import cv2
    return cv2.resize(img, tuple(POLICY_RES), interpolation=cv2.INTER_AREA)


def obs_to_frames(obs, cams):
    """Pick the requested cameras out of one sim observation (resized to --policy-res when the env renders larger
    frames for a higher-quality rollout video; the recorded video keeps the native render)."""
    if list(cams) == ["head"]:
        return _to_policy_res(obs["observation"]["head_camera"]["rgb"])       # legacy single-array form
    return {c: _to_policy_res(obs["observation"][SIM_CAM[c]]["rgb"]) for c in cams}


def obs_to_state20(obs):
    ep = obs["endpose"]
    lp, lr, lg = pose_to_state_parts(ep["left_endpose"], ep["left_gripper"])
    rp, rr, rg = pose_to_state_parts(ep["right_endpose"], ep["right_gripper"])
    return np.concatenate([lp, lr, lg, rp, rr, rg]).astype(np.float32)


def obs_to_joint14(obs):
    """[left_arm(6), left_grip(1), right_arm(6), right_grip(1)] -- the layout
    take_action(action_type='qpos') slices and the joint datasets are built in."""
    js = obs["joint_action"]
    return np.concatenate([
        np.asarray(js["left_arm"], np.float32), [np.float32(js["left_gripper"])],
        np.asarray(js["right_arm"], np.float32), [np.float32(js["right_gripper"])],
    ]).astype(np.float32)


def action_to_qpos14(a, variant, obs, cmd_prev=None):
    """Joint-space action -> the 14-vector take_action('qpos') expects.

    joint       : already an absolute target, pass through.
    joint_delta : increments on the arm joints only (the gripper command stays
                  absolute). Integrated on the previous COMMAND for the same
                  reason as the ee deltas -- re-anchoring on the measured joints
                  each step bleeds the controller's steady-state error into the
                  recursion.
    """
    a = np.asarray(a, np.float64)
    if variant.startswith("joint_delta"):
        base = np.asarray(cmd_prev if cmd_prev is not None else obs_to_joint14(obs), np.float64)
        out = base.copy()
        out[0:6] = base[0:6] + a[0:6]
        out[7:13] = base[7:13] + a[7:13]
        out[6], out[13] = a[6], a[13]          # grippers are absolute
    else:
        out = a.copy()
    out[6] = np.clip(out[6], 0.0, 1.0)
    out[13] = np.clip(out[13], 0.0, 1.0)
    return out.astype(np.float32), out


def action_to_ee16(a, variant, obs, cmd_prev=None):
    """One policy action row -> take_action 'ee' flat 16 [Lpose7,Lgrip,Rpose7,Rgrip].

    For the delta variant the increment MUST be integrated on the previously
    COMMANDED pose, not on the freshly observed one. take_action leaves a small
    steady-state tracking error eps; with obs as the base every step silently
    drops one eps (cmd_t = obs_t + d_t = cmd_{t-1} - eps + d_t) so the arm falls
    behind linearly -- measured 57.7mm after one episode, which is far more than
    the gripper tolerance. Integrating on cmd_prev keeps eps out of the
    recursion (measured 0.2mm). GT-replay: delta on obs 1/5, delta on cmd 5/5.

    Returns (ee16, cmd_now) where cmd_now feeds the next call.
    """
    ep = obs["endpose"]
    out, cmd_now = [], {}
    if variant == "absr6d":
        for ofs, side in ((0, "left"), (10, "right")):
            pos = a[ofs:ofs + 3]
            quat = rot6d_to_quat_wxyz(a[ofs + 3:ofs + 9])
            grip = np.clip(a[ofs + 9], 0.0, 1.0)
            out += [*pos, *quat, grip]
            cmd_now[side] = np.concatenate([pos, quat]).astype(np.float64)
    else:  # delta
        for ofs, side in ((0, "left"), (7, "right")):
            base = (np.asarray(cmd_prev[side], np.float64) if cmd_prev is not None
                    else np.asarray(ep[f"{side}_endpose"], np.float64))
            pos = base[:3] + a[ofs:ofs + 3]
            Rbase = R.from_quat(base[[4, 5, 6, 3]])        # wxyz -> xyzw
            Rnew = R.from_rotvec(a[ofs + 3:ofs + 6]) * Rbase
            q = Rnew.as_quat()
            grip = np.clip(a[ofs + 6], 0.0, 1.0)
            out += [*pos, q[3], q[0], q[1], q[2], grip]
            cmd_now[side] = np.array([*pos, q[3], q[0], q[1], q[2]], np.float64)
    return np.asarray(out, np.float32), cmd_now


def build_env_args(task_name, config="train_clean"):
    with open(os.path.join(CONFIGS_PATH, f"{config}.yml"), "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    emb = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    rf = emb_types[emb[0]]["file_path"]
    def emb_cfg(robot_file):
        with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)
    args["left_robot_file"] = rf
    args["right_robot_file"] = rf
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = emb_cfg(rf)
    args["right_embodiment_config"] = emb_cfg(rf)
    args["embodiment_name"] = str(emb[0])
    args["task_config"] = config
    args["save_path"] = "/tmp/robotwin_eval_scratch"   # not used for saving
    args["collect_data"] = False
    args["eval_video_log"] = False
    args["render_freq"] = 0
    return args


def make_env(task_name):
    m = importlib.import_module(f"envs.{task_name}")
    return getattr(m, task_name)()


def episode_instruction(task_name, episode_info, fallback):
    try:
        from description.utils.generate_episode_instructions import generate_episode_descriptions
        info = episode_info.get("info", {}) if isinstance(episode_info, dict) else {}
        g = generate_episode_descriptions(task_name, [info], 5)
        if g and g[0].get("seen"):
            return g[0]["seen"][0]
    except Exception as e:
        print(f"  (instruction gen failed: {e}; using fallback)")
    return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--variant", choices=["absr6d", "delta", "joint", "joint_delta"],
                    required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--replan", type=int, default=8)
    ap.add_argument("--port", type=int, default=15151)
    ap.add_argument("--seed-start", type=int, default=100000)
    ap.add_argument("--cameras", default="head",
                    help="comma list matching the checkpoint's modality video keys")
    ap.add_argument("--delta-base", default="replan", choices=["cmd", "obs", "replan"],
                    help="what a delta increment is added to: the running command "
                         "(no controller-lag bias but prediction error integrates "
                         "unboundedly), the fresh observation (feedback every step "
                         "but lag bias accumulates), or the command reset to the "
                         "observation at each replan (feedback every chunk)")
    ap.add_argument("--config", default="train_clean",
                    help="task_config yml controlling eval-time domain randomization")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixed-instr", default=None,
                    help="use this exact instruction instead of generating one (diagnostics)")
    ap.add_argument("--video-dir", default=None,
                    help="record each policy rollout (head camera) as mp4 into this dir")
    ap.add_argument("--policy-res", nargs=2, type=int, default=None, metavar=("W", "H"),
                    help="resize camera frames to this size before sending them to the policy (video keeps the env's render size)")
    ap.add_argument("--video-crf", type=int, default=26, help="x264 CRF for the rollout video (lower = better quality)")
    ap.add_argument("--skip-expert-check", action="store_true",
                    help="with --seeds: trust the list (seeds already expert-filtered elsewhere) and do not run the "
                         "expert/planner pass; lets the policy rollout run on GPUs where curobo cannot build (V100)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="evaluate exactly these seeds (skips the sequential scan)")
    ap.add_argument("--history", action="store_true",
                    help="stateful server (DP): send {'cmd':'reset'} at episode start and an "
                         "{'cmd':'obs'} update after every executed action so it can stack n_obs_steps")
    ap.add_argument("--instr-pool", default=None,
                    help="tasks.jsonl of the TRAINING dataset: sample instructions the model saw "
                         "(RoboTwin 'seen' protocol; 50-demo FT is brittle to novel paraphrases)")
    args = ap.parse_args()

    instr_pool = None
    if args.instr_pool:
        instr_pool = [json.loads(l)["task"] for l in open(args.instr_pool)][1:]  # skip index-0 fallback
        print(f"[eval] instruction pool: {len(instr_pool)} seen phrasings")
    pool_rng = np.random.default_rng(7)

    global POLICY_RES
    POLICY_RES = tuple(args.policy_res) if args.policy_res else None
    conn = socket.create_connection(("127.0.0.1", args.port))
    send_msg(conn, {"cmd": "ping"}); recv_msg(conn)
    print("[eval] server connected")

    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    env_args = build_env_args(args.task, args.config)
    fallback_instr = args.task.replace("_", " ")
    task_env = make_env(args.task)

    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)
        env_args["eval_video_save_dir"] = args.video_dir

    seed_queue = list(args.seeds) if args.seeds else None
    n_target = len(seed_queue) if seed_queue else args.n
    results, seed, tried = [], args.seed_start, 0
    while len(results) < n_target and tried < n_target * 6:
        tried += 1
        if seed_queue is not None:
            if not seed_queue:
                break
            seed = seed_queue.pop(0)
        # ---- expert check (official protocol: only score expert-solvable seeds)
        if args.skip_expert_check and seed_queue is not None:
            episode_info = None                # instruction falls back to the task name (DP ignores it)
        else:
          try:
            env_args["need_plan"] = True
            task_env.setup_demo(now_ep_num=0, seed=seed, **env_args)
            episode_info = task_env.play_once()
            ok = task_env.plan_success and task_env.check_success()
            task_env.close_env()
          except Exception as e:
            print(f"  seed {seed}: expert error {type(e).__name__}: {e}")
            task_env.close_env()
            seed += 1
            continue
          if not ok:
            print(f"  seed {seed}: expert failed, skip")
            seed += 1
            continue

        # ---- policy episode on the same seed
        if args.fixed_instr:
            instr = args.fixed_instr
        elif instr_pool is not None:
            instr = str(instr_pool[pool_rng.integers(0, len(instr_pool))])
        else:
            instr = episode_instruction(args.task, episode_info, fallback_instr)
        env_args["need_plan"] = False          # policy drives; no expert plan
        env_args["eval_mode"] = True           # loads per-task step_lim (and eval semantics)
        task_env.setup_demo(now_ep_num=len(results), seed=seed, is_test=True, **env_args)
        env_args["eval_mode"] = False          # expert check next seed runs in normal mode
        if os.environ.get("RT_STEP_LIM"):      # smoke/debug override
            task_env.step_lim = int(os.environ["RT_STEP_LIM"])
        task_env.set_instruction(instruction=instr)
        if args.video_dir:
            h, w = task_env.get_obs()["observation"]["head_camera"]["rgb"].shape[:2]
            vid_path = f"{args.video_dir}/{args.task}_{args.variant}_seed{seed}.mp4"
            ff = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                 "-pixel_format", "rgb24", "-video_size", f"{w}x{h}",
                 "-framerate", "30", "-i", "-", "-pix_fmt", "yuv420p",
                 "-vcodec", "libx264", "-crf", str(args.video_crf), vid_path],
                stdin=subprocess.PIPE)
            task_env._set_eval_video_ffmpeg(ff)
        succ = False
        try:
            cmd_prev = None
            if args.history:
                send_msg(conn, {"cmd": "reset"}); recv_msg(conn)
            while not (task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim):
                obs = task_env.get_obs()
                if args.delta_base != "cmd":
                    cmd_prev = None      # re-anchor on the observation
                send_msg(conn, {"state": obs_to_state20(obs), "joint": obs_to_joint14(obs),
                                "rgb": obs_to_frames(obs, cams),
                                "lang": instr})
                chunk = recv_msg(conn)["chunk"]
                for k in range(min(args.replan, len(chunk))):
                    if args.variant.startswith("joint"):
                        act, cmd_prev = action_to_qpos14(chunk[k], args.variant, obs, cmd_prev)
                        atype = "qpos"
                    else:
                        act, cmd_prev = action_to_ee16(chunk[k], args.variant, obs, cmd_prev)
                        atype = "ee"
                    if args.delta_base == "obs":
                        cmd_prev = None  # every step re-anchors
                    task_env.take_action(act, action_type=atype)
                    if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
                        break
                    obs = task_env.get_obs()
                    if args.history and k + 1 < min(args.replan, len(chunk)):
                        send_msg(conn, {"cmd": "obs", "joint": obs_to_joint14(obs),
                                        "rgb": obs_to_frames(obs, cams)}); recv_msg(conn)
            succ = bool(task_env.eval_success)
        except Exception as e:
            print(f"  seed {seed}: episode error {type(e).__name__}: {e}")
        if args.video_dir:
            try:
                task_env._del_eval_video_ffmpeg()
            except Exception as e:
                print(f"  (video close failed: {e})")
        task_env.close_env()
        results.append({"seed": seed, "success": succ})
        n_s = sum(r["success"] for r in results)
        print(f"[eval] ep {len(results)}/{n_target} seed={seed} success={succ}  "
              f"(SR so far {n_s}/{len(results)})", flush=True)
        seed += 1

    sr = sum(r["success"] for r in results) / max(1, len(results))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump({"task": args.task, "variant": args.variant, "replan": args.replan,
               "n": len(results), "success_rate": sr, "per_seed": results},
              open(out / "summary.json", "w"), indent=2)
    print(f"[eval] {args.task} {args.variant}: SR={sr:.2f} over {len(results)} eps -> {out}/summary.json")


if __name__ == "__main__":
    main()
