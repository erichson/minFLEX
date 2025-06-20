# ======================================================================================
#  Training driver for the super-resolution diffusion model
#  --------------------------------------------------------
#  * Distributed-Data-Parallel (DDP) setup
#  * Mixed-precision + Grad-scaler branch
#  * Exponential-Moving-Average (EMA) weights
#  * Checkpointing, sampling, and WANDB logging
# --------------------------------------------------------------------------------------
# ======================================================================================

# ---- Stdlib / third-party -------------------------------------------------------------------
import os, sys, time, numpy as np
import wandb

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import (init_process_group, destroy_process_group, barrier,
                               get_rank, get_world_size, is_initialized, all_reduce)
from torch_ema import ExponentialMovingAverage
from diffusers.optimization import get_cosine_schedule_with_warmup as scheduler
# from src.utils.lion import Lion  # optional Lion optimiser

# ---- Local project code ----------------------------------------------------------------------
from src.backbones.flex import FLEX                    # backbone definition
from src.diffusion_model_sr import DiffusionModel      # U-Net + SR diffusion model
from src.utils.get_data_sr import ERA5                 # dataset wrapper


# ════════════════════════════════════════════════════════════════════════════════════════
# 1.  DDP INITIALISATION
# ════════════════════════════════════════════════════════════════════════════════════════
def ddp_setup(local_rank: int, world_size: int) -> tuple[int, int]:
    """
    Spin up a single NCCL process group (one per GPU).

    Parameters
    ----------
    local_rank : int
        GPU index on the current *node* (0 … n_gpu-1).
    world_size : int
        Total number of participating ranks across *all* nodes.

    Returns
    -------
    local_rank : int
        Possibly updated (when launched via `torchrun` with env vars).
    global_rank : int
        Unique rank ID across the whole job.
    """
    if "MASTER_ADDR" not in os.environ:          # ── local, `mp.spawn` launch
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "3522"
        init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
        global_rank = local_rank
    else:                                        # ── multi-node or `torchrun` launch
        init_process_group(backend="nccl", init_method='env://')
        local_rank = int(os.environ["LOCAL_RANK"])   # env → int
        global_rank = get_rank()

    torch.cuda.set_device(local_rank)            # each rank sticks to a single GPU
    torch.backends.cudnn.benchmark = True        # autotune conv kernels
    return local_rank, global_rank


# ════════════════════════════════════════════════════════════════════════════════════════
# 2.  TRAINER
# ════════════════════════════════════════════════════════════════════════════════════════
class Trainer:
    """
    Encapsulates one training *driver* that lives on a **single rank**.

    Responsibilities
    ---------------
    • forward / backward pass (AMP-aware)  
    • gradient clipping & optimiser step  
    • EMA weight update  
    • periodic validation + qualitative sampling  
    • checkpoint save / restore  
    • WANDB metric emission  
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        local_gpu_id: int,
        run: wandb.wandb_sdk.wandb_run.Run,
        epochs: int,
        run_name: str,
        scratch_dir: str,
        ema_val: float = 0.999,
        clip_value: float = 1.0,
        dataset: str = 'ERA5',
        sampling_freq: int = 10,
        use_amp: bool = True,
        undo_norm=None,
    ) -> None:
        # ---- bookkeeping -------------------------------------------------------
        self.gpu_id        = gpu_id            # global rank
        self.local_gpu_id  = local_gpu_id      # device index on the node
        self.run_name      = run_name
        self.dataset       = dataset
        self.sampling_freq = sampling_freq
        self.max_epochs    = epochs
        self.undo_norm     = undo_norm

        # ---- model & optimisation ---------------------------------------------
        self.model     = model.to(local_gpu_id)
        self.model     = DDP(model, device_ids=[local_gpu_id])  # wrap before EMA
        self.ema       = ExponentialMovingAverage(model.parameters(), decay=ema_val)
        self.optimizer = optimizer
        self.clip_value = clip_value
        self.use_amp    = use_amp
        self.gscaler    = torch.cuda.amp.GradScaler() if use_amp else None

        # ---- data --------------------------------------------------------------
        self.train_data = train_data
        self.val_data   = val_data

        # ---- LR schedule -------------------------------------------------------
        self.lr_scheduler = scheduler(
            optimizer=self.optimizer,
            num_warmup_steps=len(self.train_data),            # tiny warm-up
            num_training_steps=len(self.train_data) * epochs,
        )

        # ---- logging & checkpoint ---------------------------------------------
        self.run            = run
        self.logs           = {}
        self.best_loss      = np.inf
        self.startEpoch     = 0
        self.checkpoint_dir = os.path.join(scratch_dir, 'checkpoints')
        self.checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_{dataset}_{run_name}.pt"
        )

        # ---- restore (if any) --------------------------------------------------
        if os.path.isfile(self.checkpoint_path):
            if self.gpu_id == 0:
                print(f"Loading checkpoint from {self.checkpoint_path}")
            self._restore_checkpoint(self.checkpoint_path)

    # ------------------------------------------------------------------
    # 2.1  One training epoch
    # ------------------------------------------------------------------
    def train_one_epoch(self) -> float:
        """Run a full epoch over `self.train_data`."""
        self.model.train()
        t0 = time.time()

        # fresh accumulation buffer (on GPU so we can all_reduce later)
        self.logs['train_loss'] = torch.zeros(1, device=self.local_gpu_id)

        for batch in self.train_data:
            # Move every tensor in the tuple to GPU (lowres, hires, condition, …)
            batch = [x.to(self.local_gpu_id, dtype=torch.float) for x in batch]

            self.optimizer.zero_grad(set_to_none=True)

            #  Mixed precision branch
            if self.use_amp:
                with torch.autocast(device_type="cuda"):
                    loss = self.model(*batch)
                self.gscaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                self.gscaler.step(self.optimizer)

                # Detect skipped steps → stay in warm-up a bit longer
                previous_scale = self.gscaler.get_scale()
                self.gscaler.update()
                skipped = previous_scale != self.gscaler.get_scale()
            else:
                loss = self.model(*batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                self.optimizer.step()
                skipped = False

            if not skipped:
                self.lr_scheduler.step()

            self.ema.update()                   # maintain smoothed weights
            self.logs['train_loss'] += loss.detach()

        # mean over #minibatches
        self.logs['train_loss'] /= len(self.train_data)

        # reduce across GPUs (mean-of-means)
        if is_initialized():
            all_reduce(self.logs['train_loss'])
            self.logs['train_loss'] /= get_world_size()

        return time.time() - t0

    # ------------------------------------------------------------------
    # 2.2  Validation (no grad)
    # ------------------------------------------------------------------
    def val_one_epoch(self) -> float:
        """Evaluation pass — averages loss over `self.val_data`."""
        self.model.eval()
        t0 = time.time()
        self.logs['val_loss'] = torch.zeros(1, device=self.local_gpu_id)

        with torch.no_grad():
            for batch in self.val_data:
                batch = [x.to(self.local_gpu_id, dtype=torch.float) for x in batch]
                loss = self.model(*batch)
                self.logs['val_loss'] += loss.detach()

        self.logs['val_loss'] /= len(self.val_data)

        if is_initialized():
            all_reduce(self.logs['val_loss'])
            self.logs['val_loss'] /= get_world_size()

        return time.time() - t0

    # ------------------------------------------------------------------
    # 2.3  Checkpoint helpers
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, path: str) -> None:
        """Persist model weights + optimiser & EMA states (rank-0 only)."""
        if self.gpu_id != 0:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(
            {
                'encoder':            self.model.module.encoder.state_dict(),
                'superres_encoder':   self.model.module.superres_encoder.state_dict(),
                'decoder':            self.model.module.decoder.state_dict(),
                'ema':                self.ema.state_dict(),
                'optimizer':          self.optimizer.state_dict(),
                'sched':              self.lr_scheduler.state_dict(),
                'epoch':              epoch,
                'loss':               self.best_loss,
            },
            path,
        )
        print(f"Epoch {epoch} | checkpoint saved → {path}")

    def _restore_checkpoint(self, path: str, restore_all: bool = True) -> None:
        """Load weights + state-dicts (called on every rank before training)."""
        ckpt = torch.load(path, map_location=f'cuda:{self.local_gpu_id}')

        self.model.module.encoder.load_state_dict(ckpt['encoder'])
        self.model.module.superres_encoder.load_state_dict(ckpt['superres_encoder'])
        self.model.module.decoder.load_state_dict(ckpt['decoder'])
        self.ema.load_state_dict(ckpt['ema'])

        if restore_all:
            self.startEpoch = ckpt['epoch'] + 1
            self.best_loss  = ckpt.get('loss', self.best_loss)
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.lr_scheduler.load_state_dict(ckpt['sched'])

    # ------------------------------------------------------------------
    # 2.4  Qualitative sample generation (RFNE metric)
    # ------------------------------------------------------------------
    def _generate_samples(self, epoch: int, ibreak: int = 5) -> None:
        """
        Produces a handful of SR snapshots and computes
        *Relative Frobenius-norm Error* (RFNE) for quick feedback.
        Only rank-0 writes files / logs.
        """
        self.logs['RFNE'] = torch.zeros(1, device=self.local_gpu_id)

        out_dir = f"./train_samples_{self.run_name}"
        if self.gpu_id == 0:
            os.makedirs(out_dir, exist_ok=True)

        with self.ema.average_parameters():       # swap → EMA weights
            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(self.val_data):
                    batch = [x.to(self.local_gpu_id, dtype=torch.float) for x in batch]
                    lowres, hires, Re = batch

                    # --- SR inference (diffusion sampler) ---------------------
                    preds = self.model.module.sample(
                        hires.shape[0],                          # n_sample
                        (1, hires.shape[2], hires.shape[3]),     # (C,H,W)
                        lowres,                                  # conditioning
                        Re,                                      # fluid_condition
                        device=lowres.device
                    )

                    # --- compute metric ---------------------------------------
                    preds = self.undo_norm(preds[:, 0])
                    hires = self.undo_norm(hires[:, 0])
                    rfne = torch.linalg.norm(preds - hires) / torch.linalg.norm(hires)
                    self.logs['RFNE'] += rfne.mean().detach()

                    if i >= ibreak:
                        break

        self.logs['RFNE'] /= float(ibreak)
        if is_initialized():
            all_reduce(self.logs['RFNE'])
            self.logs['RFNE'] /= get_world_size()

    # ------------------------------------------------------------------
    # 2.5  Training loop
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Main epoch loop — manages train/val, sampling, logging, ckpt."""
        for epoch in range(self.startEpoch, self.max_epochs):

            if is_initialized():
                self.train_data.sampler.set_epoch(epoch)   # reshuffle

            tic = time.time()
            tr_time  = self.train_one_epoch()
            val_time = self.val_one_epoch()

            if (epoch + 1) % self.sampling_freq == 0:
                self._generate_samples(epoch + 1)

            # ---- rank-0 console + WANDB -------------------------------------
            if self.gpu_id == 0:
                print(f"Epoch {epoch:3d} | "
                      f"train {float(self.logs['train_loss']):.4e} | "
                      f"val {float(self.logs['val_loss']):.4e} | "
                      f"learning rate {self.lr_scheduler.get_last_lr()[0]:.3e}")
                print(f"Time per epoch: {time.time() - tic:.1f}s")

                if self.run is not None:
                    self.run.log({
                        "train_loss": float(self.logs['train_loss']),
                        "val_loss":   float(self.logs['val_loss']),
                        **({"RFNE": float(self.logs['RFNE'])}
                           if 'RFNE' in self.logs else {})
                    })

            # ---- best-model checkpoint -------------------------------------
            if (self.gpu_id == 0) and (self.logs['val_loss'] < self.best_loss):
                print("  ↳ new best model — saving")
                self.best_loss = self.logs['val_loss']
                self._save_checkpoint(epoch + 1, self.checkpoint_path)


# ════════════════════════════════════════════════════════════════════════════════════════
# 3.  OBJECT FACTORIES
# ════════════════════════════════════════════════════════════════════════════════════════
def load_train_objs(args):
    """Dataset, model, optimiser factory — hides Flex/ERA5 specifics."""
    train_set = ERA5(factor=args.superres_factor, scratch_dir=args.data_dir)
    val_set   = ERA5(factor=args.superres_factor, scratch_dir=args.data_dir)

    # build UNet backbone @ 128×128 with conditional skips
    encoder, superres_encoder, _, decoder = FLEX(
        image_size=128, in_channels=1, out_channels=1,
        model_size=args.size, cond_snapshots=1
    )

    model = DiffusionModel(
        encoder=encoder.cuda(),
        decoder=decoder.cuda(),
        superres_encoder=superres_encoder.cuda(),
        n_T=args.time_steps,
        prediction_type=args.prediction_type,
        criterion=torch.nn.L1Loss()
    )

    # choose optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = Lion(model.parameters(), lr=args.learning_rate)  # alt

    return train_set, val_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    """Returns *distributed* loader (no shuffle; distribution handled by sampler)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        pin_memory=True,
        num_workers=8,
    )


# ════════════════════════════════════════════════════════════════════════════════════════
# 4.  RANK-LOCAL MAIN
# ════════════════════════════════════════════════════════════════════════════════════════
def main(rank: int, world_size: int, epochs: int, batch_size: int, run, args):
    local_rank, rank = ddp_setup(rank, world_size)
    device = torch.cuda.current_device()

    train_set, val_set, model, optimizer = load_train_objs(args)
    undo_norm = train_set.undo_norm          # for metric & sample saving

    if rank == 0:
        print('**** Model Summary ****')
        print(f'Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M')
        print('***********************')

    # DDP data loaders
    train_loader = prepare_dataloader(train_set, batch_size)
    val_loader   = prepare_dataloader(val_set, batch_size)

    trainer = Trainer(
        model, train_loader, val_loader, optimizer,
        rank, local_rank, run,
        epochs=epochs, run_name=args.run_name,
        scratch_dir=args.scratch_dir,
        dataset=args.dataset,
        sampling_freq=args.sampling_freq,
        use_amp=False,           # ← set True to enable autocast+GradScaler
        undo_norm=undo_norm,
    )

    trainer.train()
    destroy_process_group()


# ════════════════════════════════════════════════════════════════════════════════════════
# 5.  SCRIPT ENTRY-POINT
# ════════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    # ---- CLI ---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Minimalistic Diffusion Model for Super-resolution'
    )
    parser.add_argument("--run-name", type=str, default='run1')
    parser.add_argument("--dataset", type=str, default='ERA5')
    parser.add_argument("--model", type=str, default='flex')
    parser.add_argument("--size", type=str, default='small')
    parser.add_argument("--data-dir", type=str, default='data/')

    # General
    parser.add_argument("--scratch-dir", type=str, default='./')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--sampling-freq', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)

    # Super-resolution
    parser.add_argument('--superres_factor', type=int, default=8)

    # Optimiser
    parser.add_argument('--optimizer', type=str, default="lion")
    parser.add_argument('--learning-rate', type=float, default=2e-4)

    # Diffusion
    parser.add_argument("--prediction-type", type=str, default='v')
    parser.add_argument("--time-steps", type=int, default=2)

    # Multi-node
    parser.add_argument("--multi-node", action='store_true', default=False)

    args = parser.parse_args()

    # ---- Repro seed --------------------------------------------------------------------
    np.random.seed(1)

    # ---- WANDB session -----------------------------------------------------------------
    wandb.login()
    run = wandb.init(
        project="FLEX4" if args.multi_node else "FLEX",
        name=args.run_name,
        # mode="disabled",  # uncomment for offline
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch size": args.batch_size,
            "upsampling factor": args.superres_factor,
        },
    )

    # ---- Launch ­-----------------------------------------------------------------------
    if args.multi_node:
        main(0, 1, args.epochs, args.batch_size, run, args)
    else:
        world_size = torch.cuda.device_count()
        print('Launching processes...')
        mp.spawn(
            main,
            args=(world_size, args.epochs, args.batch_size, run, args),
            nprocs=world_size,
        )
        

# export MKL_THREADING_LAYER=GNU   # before you run Python
# export NCCL_ALGO=Ring           # or Tree
# export NCCL_P2P_DISABLE=1       # disables NVLink/SHARP fallback paths
# export CUDA_VISIBLE_DEVICES=0,1,2,3; python train.py --run-name flex_v_small --dataset nskt --model flex --size small --data-dir /data/rdl/NSTK/

# export CUDA_VISIBLE_DEVICES=0; python train.py --run-name flex_small

