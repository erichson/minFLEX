import torch
import torch.nn as nn
import einops
import torch.utils.checkpoint
import numpy as np
import math
from abc import abstractmethod
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
from .common import NoScaleDropout, Base2FourierFeatures, timestep_embedding, MPFourier


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x



class UpBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, embed_dim, mlp_drop, num_res_blocks,  use_checkpoint,up=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.up = up
        for _ in range(num_res_blocks):
            self.blocks.append(
                ResBlock(
                    in_channels,
                    embed_dim,
                    mlp_drop,
                    out_channels=out_channels,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=True
                )
            )
            in_channels = out_channels

        if self.up:
            self.upsample = ResBlock(in_channels,
                                     embed_dim,
                                     mlp_drop,
                                     out_channels=in_channels,
                                     use_checkpoint=use_checkpoint,
                                     use_scale_shift_norm=True,
                                     up=True,
                                     )

    def forward(self, x, emb):
        for block in self.blocks:
            x = block(x, emb)
        if self.up:
            x = self.upsample(x,emb)
        return x

    
class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv = True,  out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv=True,  out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2 
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class DownBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, embed_dim, mlp_drop, num_res_blocks,  use_checkpoint,down=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.down = down
        for _ in range(num_res_blocks):
            self.blocks.append(
                ResBlock(
                    in_channels,
                    embed_dim,
                    mlp_drop,
                    out_channels=out_channels,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=True
                )
            )
            in_channels = out_channels
        if self.down:
            self.downsample = ResBlock(in_channels,
                                       embed_dim,
                                       mlp_drop,
                                       out_channels=in_channels,
                                       use_checkpoint=use_checkpoint,
                                       use_scale_shift_norm=True,
                                       down=True,
                                       )
      

    def forward(self, x, emb):
        for block in self.blocks:
            x = block(x, emb)
        if self.down:
            x = self.downsample(x, emb)
        return x
    

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=True,
        use_scale_shift_norm=True,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d( channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """

        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)
        
    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h



def unpatchify(x, channels):
    """
    Reconstruct images from patches.

    :param x: Input tensor of shape [B, N_patches, patch_dim]
    :param channels: Number of channels in the output image
    :return: Reconstructed images of shape [B, C, H, W]
    """
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1], f"Invalid number of patches: expected {h * w}, got {x.shape[1]}"
    assert patch_size ** 2 * channels == x.shape[2], "Invalid dimensions for unpatchify"
    x = einops.rearrange(
        x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)',
        h=h, p1=patch_size, p2=patch_size
    )
    return x

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)

        qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, 'B H L D -> B L (H D)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Define layers
        self.fc1 = nn.Linear(in_features, hidden_features)
        #self.norm1 = norm_layer(hidden_features)  # LayerNorm after the first linear layer
        self.act = act_layer(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x):
        # Apply the first linear layer, activation, dropout, and norm
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        #x = self.norm1(x)

        # Apply the second linear layer, norm, and dropout
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., attn_drop=0.0, mlp_drop=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False,
                 use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop)
        self.norm2 = norm_layer(dim,  eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop, norm_layer=norm_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint


    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x,  skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x,  skip=None):
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    This module splits the image into patches and projects them to a vector space.
    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        """
        :param x: Input images of shape [B, C, H, W]
        :return: Patch embeddings of shape [B, N_patches, C]
        """
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Shape: [B, N_patches, embed_dim]
        return x


class Encoder(nn.Module):
    """
    Transformer-based U-Net model for diffusion denoising.

    This model uses a U-shaped architecture with skip connections between the encoder and decoder blocks.
    """
    def __init__(self,
                 img_size=256, 
                 in_chans=3,
                 in_conds=2,
                 model_channels = [128,256, 768],
                 num_res_blocks = [2, 2, 2, 2],
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 attn_drop=0.0, 
                 mlp_drop=0.0, 
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,         
                 use_time=True,
                 use_transf = True):
        super().__init__()

        self.in_chans = in_chans
        self.in_conds = in_conds
        self.use_time = use_time
        self.use_transf = use_transf
        self.embed_dim = model_channels[-1]
        self.extras = 1
        
        if self.use_time:
            # Time embedding module for diffusion timesteps
            self.MPFourier = MPFourier(self.embed_dim)        
            self.time_embed = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 2),
                nn.SiLU(),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
            )


        in_ch = int(model_channels[0])
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(self.in_chans, in_ch, 3, padding=1))])
        
        for level, ch in enumerate(model_channels):
            for _ in range(num_res_blocks[level]):
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            in_ch,
                            self.embed_dim,
                            mlp_drop,
                            out_channels=ch,
                            use_checkpoint=use_checkpoint,                
                        )
                    )
                )
                in_ch = ch
            
            self.input_blocks.append(
                TimestepEmbedSequential(
                    ResBlock(
                        ch,
                        self.embed_dim,
                        mlp_drop,
                        out_channels=ch,
                        use_checkpoint=use_checkpoint,
                        down=True,
                    )
                )
            )

                
        if self.in_conds > 0:
            self.cond_embed  = nn.Sequential(
                nn.Linear(self.in_conds, 2*self.embed_dim),
                nn.SiLU(),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
            )



        # Transformer
        if self.use_transf:
            # Patch embedding module
            self.patch_embed = PatchEmbed()        
            self.num_patches = (img_size //2**len(model_channels)) ** 2

            # Positional embeddings for patches and extra tokens
            self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + self.num_patches, self.embed_dim))

            # Encoder blocks (first half of the U-Net)
            self.tr_blocks = nn.ModuleList([
                Block(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
                      mlp_drop=mlp_drop, norm_layer=norm_layer, use_checkpoint=use_checkpoint)
                for _ in range(depth // 2)
            ])

            # Middle block
            self.mid_block = Block(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
                mlp_drop=mlp_drop, norm_layer=norm_layer, use_checkpoint=use_checkpoint)


        self.drop = NoScaleDropout(0.1)

        self.initialize_weights()

        
    def initialize_weights(self):        
        def _init_weights(m):
            # Initialize weights
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        if self.use_transf:
            # Initialize parameters
            nn.init.trunc_normal_(self.pos_embed, mean=0.0, std=0.02, a=-2.0, b=2.0) 
    @torch.jit.ignore
    def no_weight_decay(self):
        # Specify parameters that should not be decayed
        return {'pos_embed'}

    def forward(self, x, timesteps = None,
                fluid_condition=None, cond_skips = None):
        
        """
        Forward pass of the UViT model.

        :param x: Input images of shape [B, C_in, H, W]
        :param timesteps: Timesteps tensor of shape [B]
        :param y: Optional class labels of shape [B]
        :return: Output images of shape [B, C_out, H, W]
        """

        if fluid_condition is not None:
            # Add embedding if conditions are provided
            cond = self.cond_embed(fluid_condition)

            
        if self.use_time:
            # Create time token
            time_token = self.time_embed(self.MPFourier(timesteps))
            cond += time_token

            
        skips = []

        for layer, module in enumerate(self.input_blocks):
            x = module(x, cond)
            if cond_skips is not None:
                x = x + cond_skips[layer]
            skips.append(x)


        if self.use_transf:            
            x = self.patch_embed(x)  # Shape: [B, N_patches, embed_dim]
            cond = cond.unsqueeze(dim=1)
            x = torch.cat((cond, x), dim=1)

            B, L, D = x.shape
            
            # Add positional embeddings
            x = x + self.pos_embed
            
            # Transformers
            for blk in self.tr_blocks:
                x = blk(x)
                skips.append(x)  # Store for skip connections

            # Middle block
            x = self.mid_block(x)

        return x, skips

class Decoder(nn.Module):
    """
    Transformer-based U-Net model for diffusion denoising.

    This model uses a U-shaped architecture with skip connections between the encoder and decoder blocks.
    """
    def __init__(self,
                 img_size=256, 
                 out_chans=3,
                 in_conds = 1,
                 model_channels = [128,256, 768],
                 num_res_blocks = [2, 2, 2, 2],
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 attn_drop=0.0, 
                 mlp_drop=0.0, 
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 use_transf = False,
                 skip=True):
        super().__init__()

        self.out_chans = out_chans  # Number of output channels
        self.in_conds = in_conds
        self.extras = 1
        self.use_transf = use_transf
        self.skip = skip
        self.embed_dim = model_channels[-1]
        self.MPFourier = MPFourier(self.embed_dim)        
        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.SiLU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
        ) 

        self.patch_embed = PatchEmbed()
        if self.in_conds > 0:
            self.cond_embed  = nn.Sequential(
                nn.Linear(self.in_conds, 2*self.embed_dim),
                nn.SiLU(),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
            )



        # Decoder blocks (second half of the U-Net), with optional skip connections
        self.tr_blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
                mlp_drop=mlp_drop, norm_layer=norm_layer, skip=self.skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)
        ])
        self.norm = norm_layer(self.embed_dim)  # Final normalization layer



        ch = int(model_channels[0])
        input_block_chans = [ch]
        for level, ch in enumerate(model_channels):
            for _ in range(num_res_blocks[level]):
                input_block_chans.append(ch)
            if level != len(model_channels) - 1:
                input_block_chans.append(ch)


        #self.output_blocks = nn.ModuleList([])
        
        self.output_blocks = nn.ModuleList([TimestepEmbedSequential(
            ResBlock(
                2*ch,
                self.embed_dim,
                mlp_drop,
                out_channels=ch,
                use_checkpoint=use_checkpoint,
                up=True,
            ))])
        
        chans = input_block_chans.copy()
        for level, out_ch in list(enumerate(model_channels))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.embed_dim,
                        mlp_drop,
                        out_channels=out_ch,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                ch = out_ch
                if level and i == num_res_blocks[level]:
                    layers.append(
                        ResBlock(
                            ch,
                            self.embed_dim,
                            mlp_drop,
                            out_channels=out_ch,
                            use_checkpoint=use_checkpoint,
                            up=True,
                        )
                    )
                self.output_blocks.append(TimestepEmbedSequential(*layers))


        
        self.final_layer = nn.Sequential(
            nn.GroupNorm(32,ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, out_chans, 3, padding=1)),
        )

        self.drop = NoScaleDropout(0.1)
        # Initialize parameters
        self.initialize_weights()

    def initialize_weights(self):        
        def _init_weights( m):
            # Initialize weights
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        
    @torch.jit.ignore
    def no_weight_decay(self):
        # Specify parameters that should not be decayed
        return {'pos_embed'}

    def forward(self, x, skips,cond, cond_skips,
                timesteps,fluid_condition = None):
        
        """
        Forward pass of the UViT model.

        :param x: Input images of shape [B, C_in, H, W]
        :param timesteps: Timesteps tensor of shape [B]
        :param y: Optional class labels of shape [B]
        :return: Output images of shape [B, C_out, H, W]
        """


        time_token = self.time_embed(self.MPFourier(timesteps))
        if fluid_condition is not None:
            fluid_emb = self.cond_embed(fluid_condition)
            time_token += fluid_emb
            
        x[:,0, :] = x[:, 0, :] + time_token
            
        if self.use_transf:
            x = x + cond
            
        # Transformer
        for blk in self.tr_blocks:
            skip = skips.pop() 
            if self.use_transf:
                if self.skip:
                    skip = skip + cond_skips.pop()
                else:
                    cond_skips.pop()
                    
            x = blk(x,skip)  # Apply skip connection
            
        x = self.norm(x)
        #remove conditioning info
        x = x[:, self.extras:, :]
        
        # Decoder
        x = unpatchify(x, self.embed_dim)  # Shape: [B, C_out, H, W]
        if not self.use_transf:
            #Combine middle channel
            x = x + cond
                

        for module in self.output_blocks: 
            skip = skips.pop()+cond_skips.pop()
            x = module(torch.cat([x, skip], dim=1), time_token)

        # Final convolutional layer
        x = self.final_layer(x)  # Shape: [B, C_out, H, W]
        return x

    

def FLEX(image_size=256,
               in_channels=1,
               out_channels=1,
               model_size = 'small',
               mlp_ratio=2,
               attn_drop=0.1,
               mlp_drop=0.1,
               norm_layer=nn.LayerNorm,
               use_checkpoint=False,
               skip=True,
               use_transf = False,
               cond_snapshots = 2,
               ):


    if model_size == 'small':
        model_channels = [64, 128, 128, 256]
        decoder_res_blocks = [2, 3, 3, 3]
        encoder_res_blocks = [2, 3, 3, 3]
        depth          = 13
        num_heads      = 4

    elif model_size == 'medium':
        model_channels = [64, 128, 256, 512]
        decoder_res_blocks = [2, 3, 3, 4]
        encoder_res_blocks = [2, 3, 3, 4]
        depth          = 13
        num_heads      = 8
        

    elif model_size == 'big':
        model_channels = [128, 256, 512, 1152]
        decoder_res_blocks = [2, 3, 3, 3]
        encoder_res_blocks = [2, 3, 3, 3]
        depth          = 21
        num_heads      = 16

    else:
        raise ValueError("size not found")
    
    base_encoder =  Encoder(
        img_size=image_size,
        in_chans=in_channels,
        in_conds = 1,
        model_channels=model_channels,
        num_res_blocks = encoder_res_blocks,
        depth=depth,       
        num_heads=num_heads,    
        mlp_ratio=mlp_ratio,
        attn_drop=attn_drop,
        mlp_drop=mlp_drop,
        norm_layer=norm_layer,
        use_checkpoint=use_checkpoint,
    )

    forecast_encoder =  Encoder(
        img_size=image_size,
        in_chans=cond_snapshots,
        in_conds = 1,
        use_time = False,
        model_channels=model_channels,
        num_res_blocks = encoder_res_blocks,
        depth=depth,       
        num_heads=num_heads,    
        mlp_ratio=mlp_ratio,
        attn_drop=attn_drop,
        mlp_drop=mlp_drop,
        norm_layer=norm_layer,
        use_checkpoint=use_checkpoint,
        use_transf = use_transf,
    )

    superres_encoder =  Encoder(
        img_size=image_size,
        in_chans=in_channels,
        in_conds = 1,
        use_time = False,
        model_channels=model_channels,
        num_res_blocks = encoder_res_blocks,
        depth=depth,       
        num_heads=num_heads,    
        mlp_ratio=mlp_ratio,
        attn_drop=attn_drop,
        mlp_drop=mlp_drop,
        norm_layer=norm_layer,
        use_checkpoint=use_checkpoint,
        use_transf = use_transf,
    )

    
    
    base_decoder =  Decoder(
        img_size=image_size,
        out_chans=out_channels,
        model_channels=model_channels,
        num_res_blocks = decoder_res_blocks,
        depth=depth,       
        num_heads=num_heads,    
        mlp_ratio=mlp_ratio,
        attn_drop=attn_drop,
        mlp_drop=mlp_drop,
        norm_layer=norm_layer,
        use_checkpoint=use_checkpoint,
        use_transf = use_transf,
        skip=skip,
    )


    return base_encoder, superres_encoder, forecast_encoder, base_decoder

