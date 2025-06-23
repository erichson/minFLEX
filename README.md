.. -*- mode: rst -*-

.. image:: images/logo.png
    :width: 300px

====================================
FLEX: A Backbone for Diffusion Models
====================================

This project implements **FLEX (FLow EXpert)**, a backbone architecture for diffusion models. FLEX is a hybrid architeture, which combines convolutional ResNet layers with Transformer blocks embedded into a U-Net-style framework, optimized for tasks like super-resolving and forecasting spatio-temporal physical systems. It also supports calibrated uncertainty estimation via sampling and performs well even with as few as two reverse diffusion steps. 

The following figure illustrates the overall architecture of FLEX, instantiated for super-resolution tasks. FLEX is modular and can be extended to forecasting and multi-task settings seamlessly. Here, FLEX operates in the residual space, rather than directly modeling raw data, which stabilizes training by reducing the variance of the diffusion velocity field.

.. image:: images/flex_sr.png
    :width: 800px

See our paper on `arXiv <https://arxiv.org/abs/2505.17351>` for full details.

---------------------------
Architectural Highlights
---------------------------

- **Hybrid U-Net Backbone:**

  - Retains convolutional ResNet blocks for local spatial structure.
  - Replaces the U-Net bottleneck with a ViT (Vision Transformer) operating on patch size 1, enabling all-to-all communication without sacrificing spatial fidelity.
  - Uses a redesigned skip-connection scheme to integrate ViT bottleneck with convolutional layers, improving fine-scale reconstruction and long-range coherence.

- **Hierarchical Conditioning Strategy:**

  - Task-specific encoder processes auxiliary inputs (e.g., coarse-resolution or past snapshots).
  - Weak conditioning injects partial features via skip connections, for learnining more task-agnostic latent representation.
  - Strong conditioning injects full or learned embeddings into the decoder for task-specific guidance.


-----------------------------
Training Instructions
-----------------------------

#### Task Description

Given a coarse ERA5 snapshot of atmospheric kinetic energy density (KE), the objective is to **reconstruct** the fine-scale structure through super-resolution by a factor of **×4** or **×8**, while maintaining physical realism.

.. image:: full_image.png
    :width: 750px
    :align: center
    :alt: Full Image

Since global snapshots are too large to process directly, this tutorial focuses on super-resolving subregions. We train the model on regions of the global map and evaluate it over a selected area in the North Pacific, as indicated by the white box in the coarse-resolution snapshot above.


#### Why do we need super-resolution for climate data?

* **ERA5 is global but chunky** – a typical 0.25° grid smooths over storms, jets and valley winds.  
* **High-resolution simulations are expensive** – a year of 0.03° LES costs months on a super-computer.  
* **Downscaling bridges the gap** – we learn a mapping from *coarse* to *fine* grids, giving researchers a “microscope” for climate.



#### Environment Setup on NERSC

**1.** Start by cloning the tutorial repository to your NERSC home directory:

    git clone https://github.com/erichson/minFLEX.git

To begin model training, request an interactive GPU session on NERSC using the following command:


    salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=trn011_g


The maximum allowed time for an interactive session is 4 hours.  
Please refer to the [NERSC Interactive Job documentation](https://docs.nersc.gov/jobs/interactive/) for more detailed introduction.

Once the session starts, you can verify the assigned GPUs with:

    nvidia-smi

**2.** Load environment for training. 

Instead of creating a new virtual environment, it's recommended to use the shared PyTorch module available on the cluster:

    module load pytorch #Default version is 2.6.0
    pip install --user pytorch_ema
    pip install --user diffusers
    pip install --user eniops

Be sure to include the `--user` flag with `pip install` to avoid installing packages globally on cluster.


**3.** To train a new single-task model for super-resolution, run:

    python train.py --run-name flex_small --superres_factor 4 --prediction-type v


- The checkpoint will be saved automatically at: `checkpoints/checkpoint_ERA5_flex_small.pt`
- Weights & Biases (wandb) logging is disabled by default. To enable it, add the argument: `--use_wandb 1`
- The training script uses Distributed Data Parallel (DDP) and will automatically utilize all GPUs on the allocated node.  
  When running on 4 A100 GPUs, one epoch typically completes in approximately 50 seconds.


You can download the data and pretrained FLEX checkpoints here: `Google Drive <https://drive.google.com/drive/folders/1w3kmlXLxu6wTXmEZrX2m1R9RQGr45gTE?usp=sharing>`_.

Alternatively, data are also available on NERSC at `/global/cfs/cdirs/trn011/minFLEX`:

    minFLEX
    ├── checkpoints
    │   ├── checkpoint_ERA5_flex_small_eps_200.pt
    │   └── checkpoint_ERA5_flex_small_v_200.pt
    └── data
        └── 2013_subregion.h5

- `checkpoint_ERA5_flex_small_eps_200.pt`: model trained to predict the noise (ε)
- `checkpoint_ERA5_flex_small_v_200.pt`: model trained to predict the velocity parameter (v)
-----------------------------
Evaluation Instructions
-----------------------------

Trained models can be evaluated using the `eval.ipynb` notebook. To access a GPU-enabled Jupyter notebook on NERSC, visit: [jupyter.nersc.gov](https://jupyter.nersc.gov/hub/home)

After logging in, choose the **login node** option when prompted, and select the 'pytorch-2.6.0' **kernel** before running the notebook.