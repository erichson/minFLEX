import numpy as np
import matplotlib.pyplot as plt
import torch

import seaborn as sns
from scipy.stats import norm

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import cartopy.crs as ccrs

from src.backbones.flex import FLEX
from src.diffusion_model_sr import DiffusionModel
from torch.utils.data import Dataset, DataLoader
from torch_ema import ExponentialMovingAverage


def stich(input, res=128, target_res=256):
    X = np.zeros((target_res,target_res))
    X.shape
    
    idx = 0
    start_y = 0
    end_y = res
    for i in range(2):
        start_x = 0
        end_x = res
        for j in range(2):
            X[start_y:end_y, start_x:end_x] = input[idx, 0, :, :].reshape(res,res)
            idx += 1
            start_x += res
            end_x += res
        
        start_y += res
        end_y += res
        
    return X


def load_model(path, image_size, model_size, reverse_steps=1, prediction_type='v'):
    backbone = FLEX
    
    # pick device automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # build your model WITHOUT hard-coding .cuda()
    encoder, superres_encoder, forecast_encoder, decoder = backbone(
        image_size=128,
        in_channels=1,
        out_channels=1,
        model_size='small',
        cond_snapshots=1
    )
    
    
    model = DiffusionModel(
        encoder=encoder.to(device),
        decoder=decoder.to(device),
        superres_encoder=superres_encoder.to(device),
        n_T=reverse_steps, 
        prediction_type=prediction_type
    )
    
    # load checkpoint onto the same device
    try:
        with open(path, 'rb') as f:
            print(f'Checkpoint loaded from {path}')
            checkpoint = torch.load(f, map_location=device, weights_only=True)
            model.encoder.load_state_dict(checkpoint["encoder"])
            model.superres_encoder.load_state_dict(checkpoint["superres_encoder"])
            model.decoder.load_state_dict(checkpoint["decoder"])
            # set up EMA; the parameters are already on CPU or GPU as appropriate
            ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
            ema.load_state_dict(checkpoint["ema"])
    except (TypeError, FileNotFoundError, OSError):
        print('Loading untrained model from initialization')
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        
    # # set up EMA; the parameters are already on CPU or GPU as appropriate
    # ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    # ema.load_state_dict(checkpoint["ema"])
    
    
    # Only compile on GPU
    if device == 'cuda':
        model = torch.compile(model)
    else:
        # turn off dynamo for CPU entirely
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.disable()

    return model, ema, device




def plot_results(conditioning_snapshots, targets, dm_predictions):
    
    # 2) Build your lat/lon vectors (lat is descending, lon ascending)
    lat = np.arange( 90.0, -90.0 - 0.25, -0.25)  # [90 … -90], 721 points
    lon = np.arange(   0.0, 360.0,      0.25)   # [0 … 359.75], 1440 points
    
    # 3) Flip the latitude axis so it runs from -90…+90 (ascending)
    #    This makes slicing “northward” from lat0 just a positive index step.
    lat2     = lat[::-1]                 # now [-90 … +90], 721 points
    
    # 4) Define the SW corner and box size
    lon0, lat0 = 140.0, 10.0    # your lower-left corner
    cells      = 256            # 256×256 grid
    cell_deg   = 0.25           # degrees per cell
    span_deg   = cells * cell_deg  # 64°
    
    # 5) Find the start-indices in the “flipped” coords
    ilon0 = np.argmin(np.abs(lon  - lon0))
    ilat0 = np.argmin(np.abs(lat2 - lat0))
    
    # sanity-check the box stays in bounds
    if ilon0 + cells > lon.size or ilat0 + cells > lat2.size:
        raise IndexError("Your box would run off the grid!  Check lon0/lat0.")
    
    # 6) Extract the matching coordinate vectors
    lon_sub = lon[ ilon0:ilon0 + cells ]   # 256 long, 140→204°
    lat_sub = lat2[ilat0:ilat0 + cells ]   # 256 long, 10→74°
    
    # — your subregion coords & data —
    # lon_sub : (256,) in [0…360)
    # lat_sub : (256,) in [–90…+90] or [+90…–90]
    # conditioning_snapshots, targets, dm_predictions : each (B, C, 256, 256)
    
    # pick the first sample & the single channel
    cond  = conditioning_snapshots[0, 0, :, :]
    truth = targets               [0, 0, :, :]
    pred  = dm_predictions        [0, 0, :, :]
    
    # 1) convert lon_sub → [–180…+180)
    lon_plot_sub = (lon_sub + 180) % 360 - 180
    
    # 2) ensure lat_sub is ascending northward; if not, flip both
    lat_plot_sub = np.array(lat_sub)
    for arr in (cond, truth, pred):
        if lat_plot_sub[0] > lat_plot_sub[-1]:
            lat_plot_sub = lat_plot_sub[::-1]
            cond  = cond [::-1, :]
            truth = truth[::-1, :]
            pred  = pred [::-1, :]
    
    # 3) common vmin/vmax (so all panels share the same color‐scale)
    vmin = min(cond.min(), truth.min(), pred.min())
    vmax = max(cond.max(), truth.max(), pred.max())
    
    # 4) build 1×3 PlateCarree figure
    proj = ccrs.PlateCarree(central_longitude=180)
    fig, axs = plt.subplots(
        1, 3, figsize=(18, 6),
        subplot_kw={'projection': proj},
        constrained_layout=True
    )
    
    for ax, field, title in zip(axs, [cond, truth, pred], ["Low-res Snapshot", "Ground Truth", "FLEX Prediction"]):
        mesh = ax.pcolormesh(
            lon_plot_sub, lat_plot_sub, field,
            cmap='turbo', shading='auto',
            vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree()
        )
        ax.coastlines(resolution='50m', linewidth=1)
    
        ax.set_title(title, fontsize=14)
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.0,
            color='gray', linestyle='--'
        )
        
        gl.top_labels = gl.right_labels = False
    
        cbar = fig.colorbar(
            mesh, ax=ax, orientation='vertical',
            fraction=0.046, pad=0.04
        )
        
        cbar.set_label('Kinetic energy (m² s⁻²)')

    
    plt.show()
    
    
def plot_pull(
    dm_predictions: np.ndarray,
    targets: np.ndarray,
    sigma_pred: np.ndarray,
    *,
    bins: int = 200,
    clip: float = 6.0,
    ax: plt.Axes | None = None,
) -> tuple[float, float]:
    """
    Visualise the pull distribution for a single field (B=1, C=1 expected).

    Parameters
    ----------
    dm_predictions, targets, sigma_pred : np.ndarray
        Shape (H, W) or (1, 1, H, W).  Arrays are flattened internally.
    bins : int
        Histogram bins.
    clip : float
        Symmetric x-range [−clip, clip] shown and used for the PDF overlay.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on.  If None, a new figure is created.

    Returns
    -------
    mu_hat, sigma_hat : float, float
        Mean and width of the fitted Gaussian.
    """
    # ——— prepare data ————————————————————————————————————————————————
    err   = (dm_predictions.squeeze() - targets.squeeze()) / sigma_pred.squeeze()
    pulls = err.flatten()

    # ——— set up axes ————————————————————————————————————————————————
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    # ——— histogram ————————————————————————————————————————————————
    sns.histplot(
        pulls,
        bins=bins,
        stat="density",
        color="tab:blue",
        alpha=0.35,
        edgecolor=None,
        ax=ax,
    )

    # ——— Gaussian fit ————————————————————————————————————————————————
    mu_hat, sigma_hat = norm.fit(pulls)
    x_pdf = np.linspace(-clip, clip, 400)
    ax.plot(
        x_pdf,
        norm.pdf(x_pdf, mu_hat, sigma_hat),
        "k",
        lw=2,
        label=fr"Gaussian fit:  $\mu={mu_hat:+.2f}$,  $\sigma={sigma_hat:.2f}$",
    )

    # ——— guides & cosmetics ————————————————————————————————————————
    ax.axvline(0.0, color="black", ls="--", lw=1.0)
    ax.set_xlim(-clip, clip)
    ax.set_ylim(0, None)
    ax.set_xlabel("pull  $z = (\\hat x - x) / \\hat\\sigma$")
    ax.set_ylabel("density")
    ax.set_title("Pull distribution (calibration diagnostic)")
    ax.legend(frameon=False, loc="upper left")
    ax.figure.tight_layout()

    return float(mu_hat), float(sigma_hat)

    