import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class NoScaleDropout(nn.Module):
    """
        Dropout without rescaling.
    """
    def __init__(self, rate: float) -> None:
        super().__init__()
        self.rate = rate
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        if not self.training or self.rate == 0:
            return x
        else:
            mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = th.empty(mask_shape, device=x.device).bernoulli_(1 - self.rate)
            return x * mask

class Base2FourierFeatures(nn.Module):
    def __init__(self, start=6, stop=8, step=1):
        """
        A module to compute Base 2 Fourier Features for 2D inputs
        and append the features to the channel dimension.
        """
        super(Base2FourierFeatures, self).__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, inputs):
        """
        Args:
            inputs (th.Tensor): Input tensor of shape (B, C, H, W)
        
        Returns:
            th.Tensor: Tensor with Fourier features appended to the channel dimension.
        """
        B, C, H, W = inputs.shape  # Get the input dimensions
        freqs = th.arange(self.start, self.stop, self.step, dtype=inputs.dtype, device=inputs.device)
        w = (2.0 ** freqs) * 2 * np.pi
        w = w.repeat(C).view(1, -1, 1, 1)  # Shape (1, C * len(freqs), 1, 1)
        # Repeat and reshape inputs for Fourier computation
        inputs_expanded = inputs.repeat_interleave(len(freqs), dim=1)  # Shape (B, C * len(freqs), H, W)
        h = inputs_expanded * w  # Element-wise multiplication
        # Concatenate sine and cosine features
        fourier_features = th.cat([th.sin(h), th.cos(h)], dim=1)  # Shape (B, 2 * C * len(freqs), H, W)
        # Append the Fourier features to the original input along the channel dimension
        output = th.cat([inputs, fourier_features], dim=1)  # Shape (B, C + 2 * C * len(freqs), H, W)
        return output

class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * th.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * th.rand(num_channels))

    def forward(self, x):
        y = x.to(th.float32)
        y = y.ger(self.freqs.to(th.float32))
        y = y + self.phases.to(th.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

    
