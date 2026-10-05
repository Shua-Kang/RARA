"""Diffusion Policy training for pre-training, FT, Co-FT and RARA (Stage 2) on RoboTwin 2.0.

The loop follows RoboTwin's XPolicyLab/policy/DP RobotWorkspace, with:
  * batches from MixedBatchSampler (dataset part 0 = target, part 1 = pre-training data, fixed share per batch);
  * initialisation from a checkpoint (training.init_ckpt);
  * the RARA anchor: rows of the dataset parts in training.anchor_parts carry no behaviour-cloning loss; instead the
    encoder output on those frames is kept close to a frozen reference encoder (training.anchor_ref_ckpt):
        L = L_BC(target rows) + anchor_lambda * || f(x_ref) - f_ref(x_ref) ||^2 ;
  * light checkpoints (EMA weights + config, ~0.4 GB) loaded by eval/server.py through `load_policy`;
  * multi-GPU: `torchrun --nproc_per_node=N workspace.py ...` -> DistributedDataParallel, the global batch is split
    over ranks.
"""
import copy, json, os, pathlib, random, time
import hydra, numpy as np, torch, tqdm
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.common.pytorch_util import optimizer_to, dict_apply
from dataset import MixedRobotImageDataset, MixedBatchSampler

OmegaConf.register_new_resolver("eval", eval, replace=True)
torch.backends.cudnn.benchmark = True


def save_light_checkpoint(path, cfg, policy, epoch, global_step):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": OmegaConf.to_container(cfg, resolve=True),
                "ema_state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                "epoch": epoch, "global_step": global_step}, path)


def load_policy(path, device="cuda:0"):
    payload = torch.load(path, map_location="cpu")
    cfg = OmegaConf.create(payload["cfg"])
    policy: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["ema_state_dict"])     # includes the normalizer
    policy.to(device).eval()
    return policy, cfg, payload


def _encode(policy, encoder, obs):
    """Encoder features of the first n_obs_steps of obs (B*To, D), computed in eval mode (centre crop) so the
    trainable and the reference encoder see the same view; train mode is restored afterwards."""
    nobs = policy.normalizer.normalize(obs)
    To = policy.n_obs_steps
    this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
    was_training = encoder.training
    encoder.eval()
    try:
        return encoder(this_nobs)
    finally:
        if was_training:
            encoder.train()


class _LossModule(torch.nn.Module):
    """Behaviour cloning on the non-anchor rows + feature anchor on the anchor rows, in one forward so DDP sees every
    gradient path. forward(batch) -> tensor [total, bc, anchor] of shape (1, 3)."""

    def __init__(self, policy, ref_encoder, shares, anchor_parts, anchor_lambda):
        super().__init__()
        self.policy, self.ref_encoder = policy, ref_encoder
        offs = np.cumsum([0] + list(shares))
        parts = set(int(i) for i in anchor_parts)
        self.bc_rows = np.concatenate([np.arange(offs[i], offs[i + 1]) for i in range(len(shares)) if i not in parts])
        self.anchor_rows = np.concatenate([np.arange(offs[i], offs[i + 1]) for i in sorted(parts)]) if parts else None
        self.n_rows = int(offs[-1])
        self.anchor_lambda = float(anchor_lambda)

    def forward(self, batch):
        dev = batch["action"].device
        if len(self.bc_rows) < self.n_rows:
            rows = torch.as_tensor(self.bc_rows, device=dev)
            bc_batch = {"obs": {k: v[rows] for k, v in batch["obs"].items()}, "action": batch["action"][rows]}
        else:
            bc_batch = batch
        bc = self.policy.compute_loss(bc_batch)
        total, la = bc, torch.zeros((), device=dev)
        if self.ref_encoder is not None and self.anchor_lambda > 0 and self.anchor_rows is not None:
            rows = torch.as_tensor(self.anchor_rows, device=dev)
            obs = {k: v[rows] for k, v in batch["obs"].items()}
            with torch.no_grad():
                ref = _encode(self.policy, self.ref_encoder, obs)
            la = torch.nn.functional.mse_loss(_encode(self.policy, self.policy.obs_encoder, obs), ref)
            total = total + self.anchor_lambda * la
        return torch.stack([total, bc, la]).reshape(1, 3)


class TrainWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir or cfg.output_dir)
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_main = self.rank == 0
        if self.world_size > 1:
            torch.distributed.init_process_group("nccl")
            torch.cuda.set_device(self.local_rank)
        seed = cfg.training.seed
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
        self.ref_encoder = None
        if cfg.training.init_ckpt:
            payload = torch.load(cfg.training.init_ckpt, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(payload["ema_state_dict"], strict=False)
            if self.is_main:
                print(f"[train] init from {cfg.training.init_ckpt} (epoch {payload.get('epoch')}) "
                      f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            if self.ema_model is not None:
                self.ema_model.load_state_dict(payload["ema_state_dict"], strict=False)
        if cfg.training.anchor_lambda > 0:
            ref_src = cfg.training.anchor_ref_ckpt or cfg.training.init_ckpt
            assert ref_src, "the anchor needs training.anchor_ref_ckpt (or training.init_ckpt)"
            ref_payload = torch.load(ref_src, map_location="cpu")
            ref_policy = hydra.utils.instantiate(OmegaConf.create(ref_payload["cfg"]).policy)
            ref_policy.load_state_dict(ref_payload["ema_state_dict"])
            self.ref_encoder = copy.deepcopy(ref_policy.obs_encoder).eval().requires_grad_(False)
            if self.is_main:
                print(f"[train] anchor reference = {ref_src} (epoch {ref_payload.get('epoch')})", flush=True)
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=[p for p in self.model.parameters() if p.requires_grad])
        self.global_step, self.epoch = 0, 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        out = pathlib.Path(self.output_dir); out.mkdir(parents=True, exist_ok=True)
        if self.is_main:
            OmegaConf.save(cfg, out / "config.yaml")
        ws = self.world_size
        if ws > 1:
            assert cfg.dataloader.batch_size % ws == 0, (cfg.dataloader.batch_size, ws)
            cfg.dataloader.batch_size = cfg.dataloader.batch_size // ws      # per-rank share of the global batch
            cfg.training.device = f"cuda:{self.local_rank}"
        dataset: MixedRobotImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        sampler = MixedBatchSampler(dataset, shuffle=cfg.dataloader.shuffle, seed=cfg.training.seed, rank=self.rank, world_size=ws)
        val_dataset = dataset.get_validation_dataset()
        val_sampler = MixedBatchSampler(val_dataset, shuffle=False, seed=0)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
        n_batches = len(sampler)
        total_steps = (n_batches * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every
        if self.is_main:
            print(f"[train] parts={[len(s) for s in dataset.samplers]} shares={dataset.shares} world_size={ws} "
                  f"batches/epoch(per rank)={n_batches} epochs={cfg.training.num_epochs} total_steps={total_steps}", flush=True)
        lr_scheduler = get_scheduler(cfg.training.lr_scheduler, optimizer=self.optimizer,
                                     num_warmup_steps=cfg.training.lr_warmup_steps,
                                     num_training_steps=total_steps, last_epoch=self.global_step - 1)
        ema: EMAModel = hydra.utils.instantiate(cfg.ema, model=self.ema_model) if cfg.training.use_ema else None
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        if self.ref_encoder is not None:
            self.ref_encoder.to(device)
        optimizer_to(self.optimizer, device)

        loss_core = _LossModule(self.model, self.ref_encoder, dataset.shares, cfg.training.anchor_parts,
                                cfg.training.anchor_lambda)
        if ws > 1:
            for p_ in self.model.parameters():                 # empty placeholder parameters never receive gradients
                if p_.numel() == 0:
                    p_.requires_grad_(False)
            loss_mod = torch.nn.parallel.DistributedDataParallel(loss_core, device_ids=[self.local_rank],
                                                                 find_unused_parameters=True)
        else:
            loss_mod = loss_core

        wandb_run = None
        if self.is_main and cfg.logging.mode != "disabled":
            import wandb
            wandb_run = wandb.init(project=cfg.logging.project, entity=cfg.logging.entity, name=cfg.run_name,
                                   dir=str(out), mode=cfg.logging.mode,
                                   config=OmegaConf.to_container(cfg, resolve=True))
        jlog = open(out / "logs.json.txt", "a") if self.is_main else None

        def log(d):
            if jlog is None:
                return
            jlog.write(json.dumps(d) + "\n"); jlog.flush()
            if wandb_run is not None:
                wandb_run.log(d, step=self.global_step)

        if cfg.training.debug:
            cfg.training.num_epochs = 2; cfg.training.max_train_steps = 3; cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1; cfg.training.val_every = 1; cfg.training.sample_every = 1

        train_sampling_batch = None
        t0 = time.time()
        for _ in range(cfg.training.num_epochs):
            step_log, train_losses = {}, []
            self.model.train()
            with tqdm.tqdm(sampler, desc=f"epoch {self.epoch}", leave=False, disable=not self.is_main,
                           mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, idx in enumerate(tepoch):
                    batch = dataset.postprocess(dataset[idx], device)
                    if train_sampling_batch is None:
                        train_sampling_batch = {k: (v.clone() if torch.is_tensor(v) else {kk: vv.clone() for kk, vv in v.items()})
                                                for k, v in batch.items()}
                    outl = loss_mod(batch).reshape(-1, 3).mean(0)
                    raw_loss = outl[0]
                    (raw_loss / cfg.training.gradient_accumulate_every).backward()
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step(); self.optimizer.zero_grad(); lr_scheduler.step()
                    if ema is not None:
                        ema.step(self.model)
                    l = raw_loss.item(); train_losses.append(l)
                    tepoch.set_postfix(loss=l, refresh=False)
                    if self.is_main and self.global_step % cfg.training.log_every == 0:
                        log({"train_loss": l, "epoch": self.epoch, "lr": lr_scheduler.get_last_lr()[0],
                             "loss_bc": outl[1].item(), "loss_anchor": outl[2].item()})
                    self.global_step += 1
                    if cfg.training.max_train_steps is not None and batch_idx >= cfg.training.max_train_steps - 1:
                        break
            step_log["train_loss_epoch"] = float(np.mean(train_losses))
            step_log["epoch"] = self.epoch

            policy = self.ema_model if ema is not None else self.model
            policy.eval()
            if self.is_main:
                if self.epoch % cfg.training.val_every == 0:
                    with torch.no_grad():
                        vl = []
                        for batch_idx, idx in enumerate(val_sampler):
                            vl.append(self.model.compute_loss(val_dataset.postprocess(val_dataset[idx], device)).item())
                            if cfg.training.max_val_steps is not None and batch_idx >= cfg.training.max_val_steps - 1:
                                break
                        if vl:
                            step_log["val_loss"] = float(np.mean(vl))
                if self.epoch % cfg.training.sample_every == 0 and train_sampling_batch is not None:
                    with torch.no_grad():
                        res = policy.predict_action(train_sampling_batch["obs"])
                        step_log["train_action_mse_error"] = torch.nn.functional.mse_loss(
                            res["action_pred"], train_sampling_batch["action"]).item()
                if (self.epoch + 1) % cfg.training.checkpoint_every == 0 or self.epoch + 1 == cfg.training.num_epochs:
                    p = out / "checkpoints" / f"epoch{self.epoch + 1:04d}.pt"
                    save_light_checkpoint(p, cfg, policy, self.epoch + 1, self.global_step)
                    step_log["checkpoint"] = str(p)
                step_log["elapsed_h"] = (time.time() - t0) / 3600
                log(step_log)
                print(f"[train] epoch {self.epoch} step {self.global_step} " +
                      " ".join(f"{k}={v:.4g}" for k, v in step_log.items() if isinstance(v, float)), flush=True)
            if ws > 1:
                torch.distributed.barrier()
            self.epoch += 1
        if jlog is not None:
            jlog.close()
        if wandb_run is not None:
            wandb_run.finish()
        if ws > 1:
            torch.distributed.destroy_process_group()
        if self.is_main:
            print("[train] DONE", flush=True)


@hydra.main(version_base=None, config_path=str(pathlib.Path(__file__).parent / "config"), config_name="rara")
def main(cfg):
    TrainWorkspace(cfg).run()


if __name__ == "__main__":
    main()
