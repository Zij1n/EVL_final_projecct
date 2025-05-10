# Embodied Learning and Vision Final Project - Enhancing Visual-Motor Policies with Surface Normal Estimation

Zijin Hu, Zifan Zhao

This repo contains code for final preojct of Embodied Learning and Vision
This code is based on the following repositories:
- DynaMo: https://github.com/jeffacce/dynamo_ssl
- BYOL: https://github.com/lucidrains/byol-pytorch
- DSINE: https://github.com/baegwangbin/DSINE

## Getting started
### Setting up the project environments
- Install the project environment:
  ```
  conda env create --file=conda_env.yml
  ```
- Activate the environment:
  ```
  conda activate dynamo-repro
  ```
- To enable logging, log in with a `wandb` account:
  ```
  wandb login
  ```
  Alternatively, to disable logging altogether, set the environment variable `WANDB_MODE`:
  ```
  export WANDB_MODE=disabled
  ```

### Getting the training datasets
Please send a email for dataset access.
- Download and unzip the datasets.
- In `./configs/env_vars/env_vars.yaml`, set `dataset_root` to the unzipped parent directory containing all datasets.
- In `./eval_configs/env_vars/env_vars.yaml`, set `dataset_root` to the unzipped parent directory containing all datasets.
- In `./eval_configs/env_vars/env_vars.yaml`, set `save_path` to where you want to save the rollout results (e.g. root directory of this repo).
- Environments:
  - `sim_kitchen`: Franka kitchen environment

### Reproducing Experiments

To reproduce the experiment results, follow these steps:

1. Activate the conda environment:
   ```bash
   conda activate dynamo-repro
   ```

2. Train the visual encoder. Model snapshots will be saved to `./exp_local/...`.
   - To train the DynaMo encoder for RGB only:
     ```bash
     python3 train.py --config-name=train_sim_kitchen env.dataset.obs_type=rgb_only
     ```
   - To train the DynaMo encoder for surface normal only:
     ```bash
     python3 train.py --config-name=train_sim_kitchen env.dataset.obs_type=normal_only
     ```
   - To train the DynaMo encoder for concatenated modalities (RGB and surface normal):
     ```bash
     python3 train.py --config-name=train_sim_kitchen env.dataset.obs_type=both encoder=resnet18_more_c
     ```
   - To train the BYOL encoder:
     ```bash
     python3 train_surface_enc_byol.py 
     ```

3. Evaluate the trained encoder:
   Run the evaluation script, replacing `<CHECKPOINT PATH>` with the path to your saved model snapshot:
   ```bash
   python3 online_eval_frozen.py --config-name=train_sim_kitchen +enc_path=<CHECKPOINT PATH>
   ```

<!-- 
# **DynaMo**: In-Domain Dynamics Pretraining for Visuo-Motor Control
[[Paper]]() [[Project Website]](https://dynamo-ssl.github.io/)

[Zichen Jeff Cui](https://jeffcui.com/), [Hengkai Pan](https://www.ri.cmu.edu/ri-people/hengkai-pan/), [Aadhithya Iyer](https://aadhithya14.github.io/), [Siddhant Haldar](https://siddhanthaldar.github.io/) and [Lerrel Pinto](https://www.lerrelpinto.com/), New York University

This repo contains code for DynaMo visual pretraining, and for reproducing sim environment experiments. Datasets will be uploaded soon.


## Getting started
The following assumes our current working directory is the root directory of this project repo; tested on Ubuntu 22.04 LTS (amd64).
### Setting up the project environments
- Install the project environment:
  ```
  conda env create --file=conda_env.yml
  ```
- Activate the environment:
  ```
  conda activate dynamo-repro
  ```
- To enable logging, log in with a `wandb` account:
  ```
  wandb login
  ```
  Alternatively, to disable logging altogether, set the environment variable `WANDB_MODE`:
  ```
  export WANDB_MODE=disabled
  ```

### Getting the training datasets
Datasets used for training will be uploaded soon.
- Download and unzip the datasets.
- In `./configs/env_vars/env_vars.yaml`, set `dataset_root` to the unzipped parent directory containing all datasets.
- In `./eval_configs/env_vars/env_vars.yaml`, set `dataset_root` to the unzipped parent directory containing all datasets.
- In `./eval_configs/env_vars/env_vars.yaml`, set `save_path` to where you want to save the rollout results (e.g. root directory of this repo).
- Environments:
  - `sim_kitchen`: Franka kitchen environment
  - `block_push_multiview`: Block push environment
  - `libero_goal`: LIBERO Goal environment
  - `pusht`: Push-T environment

## Reproducing experiments
The following assumes our current working directory is the root directory of this project repo.

To reproduce the experiment results, the overall steps are:
1. Activate the conda environment with
   ```
   conda activate dynamo-repro
   ```

2. Train the visual encoder with `python3 train.py --config-name=train_*`. A model snapshot will be saved to `./exp_local/...`;
3. In `eval_configs/encoder`, in the corresponding environment config, set the encoder file path `f` to the saved snapshot;
4. Eval with `python3 online_eval.py --config-name=train_*`.

See below for detailed steps for each environment.


### Franka Kitchen
- Train the encoder:
  ```
  python3 train.py --config-name=train_sim_kitchen
  ```
  Snapshots will be saved to a new timestamped directory `./exp_local/{date}/{time}_train_sim_kitchen_dynamo`.

  The encoder snapshot will be at `./exp_local/{date}/{time}_train_sim_kitchen_dynamo/encoder.pt`.
- In `eval_configs/encoder/kitchen_dynamo.yaml`, set `SNAPSHOT_PATH` to the absolute path of the encoder snapshot above.
- Evaluation:
  ```
  MUJOCO_GL=egl python3 online_eval.py --config-name=train_sim_kitchen
  ```

### Block Pushing
- Train the encoder:
  ```
  python3 train.py --config-name=train_blockpush
  ```
  Snapshots will be saved to a new timestamped directory `./exp_local/{date}/{time}_train_blockpush_dynamo`.

  The encoder snapshot will be at `./exp_local/{date}/{time}_train_blockpush_dynamo/encoder.pt`.
- In `eval_configs/encoder/blockpush_dynamo.yaml`, set `SNAPSHOT_PATH` to the absolute path of the encoder snapshot above.
- Evaluation:
  ```
  ASSET_PATH=$(pwd) python3 online_eval.py --config-name=train_blockpush
  ```
  (Evaluation requires including this repository in `ASSET_PATH`.)

### Push-T
- Train:
  ```
  python3 train.py --config-name=train_pusht
  ```
  Snapshots will be saved to a new timestamped directory `./exp_local/{date}/{time}_train_pusht_dynamo`.
  
  The encoder snapshot will be at `./exp_local/{date}/{time}_train_pusht_dynamo/encoder.pt`
- In `eval_configs/encoder/pusht_dynamo.yaml`, set `SNAPSHOT_PATH` to the absolute path of the encoder snapshot above.
- Evaluation:
  ```
  python3 online_eval.py --config-name=train_pusht
  ```

### LIBERO Goal
- Train:
  ```
  python3 train.py --config-name=train_libero_goal
  ```
  Snapshots will be saved to a new timestamped directory `./exp_local/{date}/{time}_train_libero_goal_dynamo`.

  The encoder snapshot will be at `./exp_local/{date}/{time}_train_libero_goal_dynamo/encoder.pt`
- In `eval_configs/encoder/libero_dynamo.yaml`, set `SNAPSHOT_PATH` to the absolute path of the encoder snapshot above.
- Evaluation:
  ```
  MUJOCO_GL=egl python3 online_eval.py --config-name=train_libero_goal
  ```

## Train on your own dataset
- Plug in your dataset in these files:
  - `datasets/your_dataset.py`
  - `configs/env/your_dataset.yaml`
  - `configs/env_vars/env_vars.yaml`

- Check the inverse/forward model configs:
  - `configs/train_your_dataset.yaml`
    - This is the main config.
  - `configs/ssl/dynamo_your_dataset.yaml`
    - If the model converges slowly, try setting `ema_beta` to `null` to use SimSiam instead of EMA encoder during training.
  - `configs/projector/inverse_dynamics_your_dataset.yaml`
    - We find that setting the inverse dynamics `output_dim` to approximately the underlying state dimension usually works well.
      - For sim environments, this is the state-based observation dimension.
      - For real environments, e.g. a 7DoF robot arm manipulating a rigid object (6D), this would be ~16 dimensions.

- Add linear probes for training diagnostics:
  - `workspaces/your_workspace.py`
    - This template computes linear probe and nearest neighbor MSE from the image embeddings to states/actions, for monitoring training convergence.
    - It assumes that your dataset class has `states` (`batch` x `time` x `state_dim`) and `actions` (`batch` x `time` x `action_dim`) attributes.
      - For a real-world dataset, you can use proprioception as the state. -->
