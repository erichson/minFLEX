"""
Created on Fri Jul  5, 2024

@author: ben
"""


from __future__ import annotations  # enables |‐based type unions on Python <3.10

import copy  # NOTE: not used at present, but kept in case it is required downstream
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401  # kept for potential future use

# --------------------------------------------------------------------------------------
# Log‑SNR schedule helpers
# --------------------------------------------------------------------------------------


def logsnr_schedule_cosine(
    t: torch.Tensor,
    logsnr_min: float = -20.0,
    logsnr_max: float = 20.0,
    shift: float = 1.0,
) -> torch.Tensor:
    """Cosine log‑SNR schedule from *Nichol & Dhariwal (2021)* with optional *shift*.

    This schedule maps a *normalised* continuous time‐step *t∈[0,1]* to the **log‑signal‑
    to‑noise‑ratio** (log‑SNR) used in diffusion models.  The closed‑form expression below
    is derived from the original paper; the *shift* parameter is retained solely for
    backward‑compatibility with older checkpoints – setting it to values ≠ 1.0 will *scale*
    the resulting schedule but will **not** change its shape.

    Args:
        t: Normalised time in **[0, 1]**; arbitrary leading dimensions are allowed, but
           shape *(batch,)* is most common.
        logsnr_min / logsnr_max: Lower/upper bounds of the log‑SNR range.
        shift: Scalar multiplier that uniformly shifts the curve along the vertical axis.

    Returns:
        logsnr: Tensor with the same leading dimensions as *t* containing log‑SNR values.
    """
    # The transformation below is numerically stable for the specified default range.
    b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
    a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - b
    return -2.0 * torch.log(torch.tan(a * t + b) * shift)


# Wrapper that additionally provides *α* and *σ* — the square‑rooted signal/noise weights
# used by most modern parameterisations (x₀, ε, or v) -----------------------------------

def get_logsnr_alpha_sigma(
    time: torch.Tensor,
    shift: float = 16.0,  # NOTE: 16.0 follows the "imagen" implementation; set to 1.0 to match
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(logsnr, α, σ)` broadcastable to **(B, 1, 1, 1)**.

    The helper expands the 1‑D *time* tensor into per‑sample log‑SNR plus its derived
    coefficients *α* and *σ* such that subsequent point‑wise arithmetic broadcasts cleanly
    over spatial dimensions.
    """
    logsnr = logsnr_schedule_cosine(time, shift=shift)[:, None, None, None]
    alpha = torch.sqrt(torch.sigmoid(logsnr))       # α = (SNR / (1+SNR))^½
    sigma = torch.sqrt(torch.sigmoid(-logsnr))      # σ = ( 1  / (1+SNR))^½
    return logsnr, alpha, sigma


# --------------------------------------------------------------------------------------
# Diffusion model – U‑Net backbone + super‑resolution branch
# --------------------------------------------------------------------------------------


class DiffusionModel(nn.Module):
    """U‑Net‑style *conditional* diffusion model with an auxiliary super‑resolution head.

    The network is conceptually split into three parts:

    1. **Super‑resolution encoder** – extracts low‑resolution features (skip connections)
       from the input *conditioning* frames.
    2. **Main encoder (U‑Net)** – processes the *noisy* residual (xₜ) together with the
       conditioning features and the diffusion timestep *t*.
    3. **Decoder** – merges both streams and predicts either **x₀**, **ε**, or **v**
       depending on *prediction_type*.

    Notes
    -----
    * All conditioning inputs (fluid flow, etc.) are threaded through identically so that
      the model can learn physics‑aware super‑resolution.
    * The "velocity" parameterisation (v) generally yields better‑behaved gradients, but
      we preserve backwards‑compatibility with the other two for checkpoint reuse.
    """

    # ------------------------------------------------------------------
    # Instantiation helpers
    # ------------------------------------------------------------------

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        superres_encoder: nn.Module,
        n_T: int,
        prediction_type: str,
        criterion: nn.Module | None = None,
        logsnr_shift: float = 1.0,
    ) -> None:
        super().__init__()

        assert prediction_type in {"v", "eps", "x"}, (
            "prediction_type must be one of 'v', 'eps', 'x'"
        )
        self.prediction_type = prediction_type

        # Sub‑modules ----------------------------------------------------------------
        self.encoder = encoder
        self.decoder = decoder
        self.superres_encoder = superres_encoder

        # Hyper‑parameters -----------------------------------------------------------
        self.n_T = n_T  # total number of diffusion steps during *sampling*
        # If *criterion* is omitted we default to a per‑pixel L1 loss ("p‑loss")
        self.criterion = criterion or nn.L1Loss(reduction="none")
        self.logsnr_shift = logsnr_shift  # shift passed to get_logsnr_alpha_sigma

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        lowres_snapshots: torch.Tensor,
        snapshots: torch.Tensor,
        fluid_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Compute **per‑pixel loss** for a *single random* diffusion timestep.

        The routine implements the standard diffusion training recipe:

        1. **Sample** a random timestep *t*.
        2. **Diffuse** the target residual x₀ → xₜ using Gaussian noise ε.
        3. **Predict** a target (x₀/ε/v) with the network.
        4. **Return** the per‑pixel loss so callers can decide on the reduction.
        """

        # ---------------------------------------------------------------------------
        # 0. Ground‑truth residual between full‑res and low‑res frames ----------------
        # ---------------------------------------------------------------------------
        residual_sr = snapshots - lowres_snapshots  # x₀  (B,C,H,W)

        # ---------------------------------------------------------------------------
        # 1. Random timestep t ∼ 𝕌(0,1) and corresponding schedule coefficients -------
        # ---------------------------------------------------------------------------
        t = torch.rand(residual_sr.shape[0], device=residual_sr.device)
        logsnr, alpha, sigma = get_logsnr_alpha_sigma(t, shift=self.logsnr_shift)

        # ---------------------------------------------------------------------------
        # 2. Forward diffusion (add Gaussian noise) ----------------------------------
        # ---------------------------------------------------------------------------
        eps = torch.randn_like(residual_sr)                       # ε ∼ 𝒩(0, I)
        residual_t = alpha * residual_sr + sigma * eps            # xₜ

        # ---------------------------------------------------------------------------
        # 3. Neural‑net forward pass -------------------------------------------------
        # ---------------------------------------------------------------------------

        # 3.1 Low‑resolution conditioning path (provides skip connections) ----------
        head_sr, skips_sr = self.superres_encoder(
            lowres_snapshots, fluid_condition=fluid_condition
        )

        # 3.2 Diffusion path (main U‑Net) -------------------------------------------
        h, skips = self.encoder(
            residual_t, t, fluid_condition=fluid_condition, cond_skips=skips_sr
        )

        # 3.3 Decoder merges streams + timestep embedding ---------------------------
        pred = self.decoder(
            h, skips, head_sr, skips_sr, t, fluid_condition=fluid_condition
        )

        # ---------------------------------------------------------------------------
        # 4. Convert *pred* to the correct target space -----------------------------
        # ---------------------------------------------------------------------------
        if self.prediction_type == "x":
            target = residual_sr                           # x₀ (direct regression)

        elif self.prediction_type == "eps":
            # Network is trained as *v* but we supervise with ε.
            pred = alpha * pred + sigma * residual_t       # ε̂ (predicted)
            target = eps                                   # ε  (ground‑truth)

        elif self.prediction_type == "v":
            # Velocity parameterisation: v = α ε − σ x₀
            target = alpha * eps - sigma * residual_sr

        # The criterion is reduction="none" by default, so the caller retains control
        # (e.g. they can add importance weighting or reduce to *mean* later).
        return self.criterion(pred, target)

    # ------------------------------------------------------------------
    # DDPM/DDIM‑like sampler (iterative denoising)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        n_sample: int,
        size: Tuple[int, int, int],  # (C, H, W)
        conditioning_snapshots: torch.Tensor,
        fluid_condition: torch.Tensor,
        device: str = "cuda",
        superres: bool = False,  # NOTE: currently unused – could toggle branch execution
        snapshots_i: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Iterative *reverse* diffusion from *t = 1* → 0* ala DDPM/DDIM.

        Args:
            n_sample: Number of samples to generate.
            size: Output spatial size as *(C, H, W)*.
            conditioning_snapshots: Low‑resolution conditioning frames.
            fluid_condition: Auxiliary physical fields (e.g. velocity, vorticity).
            device: Device to run on (default: "cuda").
            superres: Placeholder flag – not yet used.
            snapshots_i: Optional initial noise tensor; if *None* a fresh *x_T* is drawn.

        Returns:
            A tensor of shape *(n_sample, C, H, W)* containing the generated snapshots.
        """

        # -----------------------------------------------------------------------
        # 0. Initialise with Gaussian noise (or user‑supplied *x_T*)
        # -----------------------------------------------------------------------
        if snapshots_i is None:
            snapshots_i = torch.randn(n_sample, *size, device=device)  # x_T

        # Conditioning tensor (low‑res video, last frame etc.) ---------------------
        conditional = conditioning_snapshots.to(device)
        model_head = self.superres_encoder  # alias for brevity

        # -----------------------------------------------------------------------
        # 1. Reverse diffusion loop t = 1 → 0
        # -----------------------------------------------------------------------
        for time_step in range(self.n_T, 0, -1):
            # Current and previous (t‑1) timesteps normalised to [0,1]
            t  = torch.full((n_sample,),  time_step / self.n_T,  device=device)
            t_ = torch.full((n_sample,), (time_step - 1) / self.n_T, device=device)

            _ ,  alpha,  sigma  = get_logsnr_alpha_sigma(t,  shift=self.logsnr_shift)
            _ , alpha_, sigma_ = get_logsnr_alpha_sigma(t_, shift=self.logsnr_shift)

            # ---------------------------
            # 1.1 Conditioning encoder --
            # ---------------------------
            pred_head, skip_head = model_head(conditional, fluid_condition=fluid_condition)

            # ---------------------------
            # 1.2 Main U‑Net forward ----
            # ---------------------------
            h, skip = self.encoder(
                snapshots_i, t, fluid_condition=fluid_condition, cond_skips=skip_head
            )
            pred = self.decoder(
                h, skip, pred_head, skip_head, t, fluid_condition=fluid_condition
            )

            # -----------------------------------------------------------
            # 1.3 Convert network output to (mean, eps) pair ----------
            # -----------------------------------------------------------
            if self.prediction_type == "v":
                mean = alpha * snapshots_i - sigma * pred
                eps  = alpha * pred       + sigma * snapshots_i

            elif self.prediction_type == "x":
                mean = pred  # x₀ (direct prediction)
                eps  = (alpha * pred - snapshots_i) / sigma

            elif self.prediction_type == "eps":
                mean = alpha * snapshots_i - sigma * pred  # identical to 'v'
                eps  = alpha * pred       + sigma * snapshots_i

            # -----------------------------------------------------------
            # 1.4 DDIM update (deterministic if η = 0) ----------------
            # -----------------------------------------------------------
            eta = 0.0  # 0 → DDIM   |   1 → DDPM (full noise)
            noise = torch.randn_like(snapshots_i) if eta > 0 else 0.0
            snapshots_i = alpha_ * mean + sigma_ * eps + eta * sigma_ * noise

        # -----------------------------------------------------------------------
        # 2. Final prediction uses *mean* (deterministic)
        # -----------------------------------------------------------------------
        snapshots_i = mean  # last mean corresponds to t=0

        # -----------------------------------------------------------------------
        # 3. Add low‑res conditioning back to obtain full‑res prediction
        # -----------------------------------------------------------------------
        # If the conditioning tensor is a *sequence*, we only add the last frame.
        if conditional.shape[1] > 1:
            conditional = conditional[:, -1:, ...]
        return snapshots_i + conditional