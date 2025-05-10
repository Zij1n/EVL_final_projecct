import einops
import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

import wandb
from utils.video import VideoRecorder
import pickle
from datasets.core import TrajectoryEmbeddingDataset, split_traj_datasets
from datasets.vqbet_repro import TrajectorySlicerDataset
from models.encoder.resnet_w_decoder import (
    resnet18,
    angular_loss,
)  # Import the model and loss function


if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


@hydra.main(config_path="eval_configs", version_base="1.2")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    # Define camera intrinsics
    intrinsics = None

    encoder = hydra.utils.instantiate(cfg.encoder)
    print(encoder)
    # Ensure the encoder is the correct type for compute_loss
    # This assumes cfg.encoder instantiates the resnet18 model from resnet_w_decoder
    assert isinstance(encoder, resnet18), (
        "Encoder must be an instance of resnet18 from resnet_w_decoder.py to compute auxiliary loss"
    )
    encoder = encoder.to(
        cfg.device
    )  # Keep in eval mode initially, switch in loop if not frozen

    dataset = hydra.utils.instantiate(cfg.dataset)
    print(dataset)
    train_data, test_data = split_traj_datasets(
        dataset,
        train_fraction=cfg.train_fraction,
        random_seed=cfg.seed,
    )
    use_libero_goal = cfg.data.get("use_libero_goal", False)
    # train_data = TrajectoryEmbeddingDataset(
    #     encoder, train_data, device=cfg.device, embed_goal=use_libero_goal
    # )
    # test_data = TrajectoryEmbeddingDataset(
    #     encoder, test_data, device=cfg.device, embed_goal=use_libero_goal
    # )
    traj_slicer_kwargs = {
        "window": cfg.data.window_size,
        "action_window": cfg.data.action_window_size,
        "vqbet_get_future_action_chunk": cfg.data.vqbet_get_future_action_chunk,
        "future_conditional": (cfg.data.goal_conditional == "future"),
        "min_future_sep": cfg.data.action_window_size,
        "future_seq_len": cfg.data.future_seq_len,
        "use_libero_goal": use_libero_goal,
    }
    train_data = TrajectorySlicerDataset(train_data, **traj_slicer_kwargs)
    test_data = TrajectorySlicerDataset(test_data, **traj_slicer_kwargs)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False, pin_memory=False
    )
    if cfg.frozen_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen!!!!")
        encoder.eval()
    else:
        trainable_params = sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        )
        print(f"Number of trainable parameters: {trainable_params}")
        encoder.train()

    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=cfg.optim.betas,
    )

    optimizer.add_param_group(
        {
            "params": encoder.parameters(),
            "weight_decay": cfg.optim.weight_decay,
            "lr": cfg.optim.lr,
            "betas": cfg.optim.betas,
        }
    )
    for i, g in enumerate(optimizer.param_groups):
        print(f"group {i}: {len(g['params'])} params, lr={g['lr']}")

    total_params = sum(
        p.numel()
        for group in optimizer.param_groups
        for p in group["params"]
        if p.requires_grad
    )
    print(f"Total ob from optimizer: {total_params}")

    env = hydra.utils.instantiate(cfg.env.gym)
    if "use_libero_goal" in cfg.data:
        with torch.no_grad():
            # calculate goal embeddings for each task
            goals_cache = []
            for i in range(10):
                idx = i * 50
                last_obs, _, _ = dataset.get_frames(idx, [-1])  # 1 V C H W
                last_obs = last_obs.to(cfg.device)
                embd = encoder(last_obs)[0]  # V E
                embd = einops.rearrange(embd, "V E -> (V E)")
                goals_cache.append(embd)

        def goal_fn(goal_idx):
            return goals_cache[goal_idx]
    else:
        empty_tensor = torch.zeros(1)

        def goal_fn(goal_idx):
            return empty_tensor

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run_name = run.name or "Offline"
    run.watch(encoder)
    save_path = Path(cfg.save_path) / run_name
    save_path.mkdir(parents=True, exist_ok=False)
    video = VideoRecorder(dir_name=save_path)

    @torch.no_grad()
    def eval_on_env(
        cfg,
        num_evals=cfg.num_env_evals,
        num_eval_per_goal=1,
        videorecorder=None,
        epoch=None,
    ):
        include_surface_normal = True
        if include_surface_normal:
            from models.encoder.surface_normal import DSINE

            normal_estimator = DSINE()

        def embed(enc, obs, include_surface_normal=True):
            obs = (
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(cfg.device)
            )  # 1 V C H W
            if include_surface_normal:
                normals = normal_estimator(obs)
                obs = torch.cat((obs, normals), dim=2)
            result = enc(obs)
            result = einops.rearrange(result, "1 V E -> (V E)")
            return result

        avg_reward = 0
        action_list = []
        completion_id_list = []
        avg_max_coverage = []
        avg_final_coverage = []
        env.seed(cfg.seed)
        for goal_idx in tqdm.tqdm(range(num_evals)):
            if videorecorder is not None:
                videorecorder.init(enabled=True)
            for i in range(num_eval_per_goal):
                obs_stack = deque(maxlen=cfg.eval_window_size)
                this_obs = env.reset(goal_idx=goal_idx)  # V C H W
                assert this_obs.min() >= 0 and this_obs.max() <= 1, (
                    "expect 0-1 range observation"
                )
                this_obs_enc = embed(encoder, this_obs)
                obs_stack.append(this_obs_enc)
                done, step, total_reward = False, 0, 0
                goal = goal_fn(goal_idx)  # V C H W
                while not done:
                    obs = torch.stack(tuple(obs_stack)).float().to(cfg.device)
                    goal = torch.as_tensor(goal, dtype=torch.float32, device=cfg.device)
                    # goal = embed(encoder, goal)
                    goal = goal.unsqueeze(0).repeat(cfg.eval_window_size, 1)
                    action, _, _ = cbet_model(obs.unsqueeze(0), goal.unsqueeze(0), None)
                    action = action[0]  # remove batch dim; always 1
                    if cfg.action_window_size > 1:
                        action_list.append(action[-1].cpu().detach().numpy())
                        if len(action_list) > cfg.action_window_size:
                            action_list = action_list[1:]
                        curr_action = np.array(action_list)
                        curr_action = (
                            np.sum(curr_action, axis=0)[0] / curr_action.shape[0]
                        )
                        new_action_list = []
                        for a_chunk in action_list:
                            new_action_list.append(
                                np.concatenate(
                                    (a_chunk[1:], np.zeros((1, a_chunk.shape[1])))
                                )
                            )
                        action_list = new_action_list
                    else:
                        curr_action = action[-1, 0, :].cpu().detach().numpy()

                    this_obs, reward, done, info = env.step(curr_action)
                    this_obs_enc = embed(encoder, this_obs)
                    obs_stack.append(this_obs_enc)

                    if videorecorder.enabled:
                        videorecorder.record(info["image"])
                    step += 1
                    total_reward += reward
                    goal = goal_fn(goal_idx)
                avg_reward += total_reward
                if cfg.env.gym.id == "pusht":
                    env.env._seed += 1
                    avg_max_coverage.append(info["max_coverage"])
                    avg_final_coverage.append(info["final_coverage"])
                elif cfg.env.gym.id == "blockpush":
                    avg_max_coverage.append(info["moved"])
                    avg_final_coverage.append(info["entered"])
                completion_id_list.append(info["all_completions_ids"])
            videorecorder.save("eval_{}_{}.mp4".format(epoch, goal_idx))
        return (
            avg_reward / (num_evals * num_eval_per_goal),
            completion_id_list,
            avg_max_coverage,
            avg_final_coverage,
        )

    metrics_history = []
    reward_history = []
    for epoch in tqdm.trange(cfg.epochs):
        cbet_model.eval()
        if epoch % cfg.eval_on_env_freq == 0 and epoch != 0:
            avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
                cfg,
                videorecorder=video,
                epoch=epoch,
                num_eval_per_goal=cfg.num_final_eval_per_goal,
            )
            reward_history.append(avg_reward)
            with open("{}/completion_idx_{}.json".format(save_path, epoch), "wb") as fp:
                pickle.dump(completion_id_list, fp)
            wandb.log({"eval_on_env": avg_reward})
            if cfg.env.gym.id in ["pusht", "blockpush"]:
                metric_final = (
                    "final coverage" if cfg.env.gym.id == "pusht" else "entered"
                )
                metric_max = "max coverage" if cfg.env.gym.id == "pusht" else "moved"
                metrics = {
                    f"{metric_final} mean": sum(final_coverage) / len(final_coverage),
                    f"{metric_final} max": max(final_coverage),
                    f"{metric_final} min": min(final_coverage),
                    f"{metric_max} mean": sum(max_coverage) / len(max_coverage),
                    f"{metric_max} max": max(max_coverage),
                    f"{metric_max} min": min(max_coverage),
                }
                wandb.log(metrics)
                metrics_history.append(metrics)

        if epoch % cfg.eval_freq == 0:
            total_loss = 0
            action_diff = 0
            action_diff_tot = 0
            action_diff_mean_res1 = 0
            action_diff_mean_res2 = 0
            action_diff_max = 0
            with torch.no_grad():
                import time

                loader_iter = iter(test_loader)

                for i in range(len(test_loader)):
                    # t0 = time.perf_counter()
                    data = next(loader_iter)  # this is the actual data loading
                    # t1 = time.perf_counter()

                    raw_obs, act, goal = (x.to(cfg.device) for x in data)

                    # --- Separate RGB for Embedding ---
                    # Assuming raw_obs has shape (N, T, V, C, H, W) where C=6 (3 RGB + 3 Normal)
                    # print(raw_obs.shape)
                    B, N, V, C, H, W = raw_obs.shape
                    assert C == 6, (
                        f"Expected 6 channels (RGB+Normal) in observation, got {C}"
                    )

                    img_obs = raw_obs[..., :3, :, :]  # (N, T, V, 3, H, W)

                    # t2 = time.perf_counter()
                    # Pass only RGB channels to the encoder for the main task embedding
                    obs_encoded = encoder(img_obs)
                    assert obs_encoded.ndim == 4, "expect N T V E here"
                    obs_encoded = einops.rearrange(obs_encoded, "N T V E -> N T (V E)")
                    goal = einops.rearrange(
                        goal, "N T V E -> N T (V E)"
                    )  # Assuming goal is already encoded or doesn't need it

                    predicted_act, loss, loss_dict = cbet_model(obs_encoded, goal, act)
                    total_loss += loss.item()
                    wandb.log({"eval/{}".format(x): y for (x, y) in loss_dict.items()})
                    action_diff += loss_dict["action_diff"]
                    action_diff_tot += loss_dict["action_diff_tot"]
                    action_diff_mean_res1 += loss_dict["action_diff_mean_res1"]
                    action_diff_mean_res2 += loss_dict["action_diff_mean_res2"]
                    action_diff_max += loss_dict["action_diff_max"]
                    # t3 = time.perf_counter()

                    # print(
                    #     f"[{i}] Data load: {t1 - t0:.4f}s | To device: {t2 - t1:.4f}s | Compute: {t3 - t2:.4f}s"
                    # )
            print(f"Test loss: {total_loss / len(test_loader)}")
            wandb.log({"eval/epoch_wise_action_diff": action_diff})
            wandb.log({"eval/epoch_wise_action_diff_tot": action_diff_tot})
            wandb.log({"eval/epoch_wise_action_diff_mean_res1": action_diff_mean_res1})
            wandb.log({"eval/epoch_wise_action_diff_mean_res2": action_diff_mean_res2})
            wandb.log({"eval/epoch_wise_action_diff_max": action_diff_max})

        cbet_model.train()
        encoder.train()
        # Define auxiliary loss weight (consider adding this to cfg)
        aux_loss_weight = cfg.aux_loss_weight
        print
        for data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            # raw_obs shape from DataLoader: N, V, C, H, W where N is batch/time dimension
            raw_obs, act, goal = (x.to(cfg.device) for x in data)

            # --- Auxiliary Loss Calculation ---
            # Assuming raw_obs has shape (N, V, C, H, W) where N is batch/time and C=6 (3 RGB + 3 Normal)
            B, N, V, C, H, W = raw_obs.shape
            assert C == 6, f"Expected 6 channels (RGB+Normal) in observation, got {C}"

            img_obs = raw_obs[..., :3, :, :]  # (N, V, 3, H, W)
            gt_norm = raw_obs[..., 3:, :, :]  # (N, V, 3, H, W)

            # Reshape for compute_loss: expects (B, [V], C, H, W)
            # Combine N and V into batch dimension B = N * V
            img_obs_flat = img_obs.reshape(B * N * V, 3, H, W)
            gt_norm_flat = gt_norm.reshape(B * N * V, 3, H, W)

            # Create mask (assuming all GT normals are valid)
            gt_norm_mask_flat = torch.ones(
                B * N * V, 1, H, W, dtype=torch.bool, device=cfg.device
            )

            # Prepare intrinsics: repeat for batch dimension (N * V)
            # Expected shape: (B, [V], 3, 3) -> (N * V, 1, 3, 3) - assuming V is handled by compute_loss
            # Or (N * V, 3, 3) if compute_loss expects B, C, H, W and handles V internally
            # Let's assume compute_loss expects (B, [V], C, H, W) and intrinsics (B, [V], 3, 3)
            # So intrinsics need to be (N, V, 3, 3) repeated for the flat batch
            intrins_flat = None

            # Calculate auxiliary loss using the encoder's method
            # Ensure encoder is in train mode if not frozen
            # if not cfg.frozen_encoder:
            encoder.train()  # Make sure encoder is in train mode for aux loss calculation if not frozen
            aux_loss = encoder.compute_loss(
                img_obs_flat, gt_norm_flat, gt_norm_mask_flat, intrins_flat
            )
            # if cfg.frozen_encoder:  # Switch back to eval if it was frozen
            #     encoder.eval()

            # --- Primary CBET Loss Calculation ---
            # Encode the image part of the observation for CBET
            # Input to encoder is (N, V, 3, H, W)
            # Output from encoder is (N, V, E)
            obs_encoded = encoder(img_obs)

            # Reshape for cbet_model: expects (Batch_Size, T, (V*E))
            # Assuming N is both Batch_Size and T (time)
            assert obs_encoded.ndim == 4, "expect N T V E here"
            obs_encoded = einops.rearrange(obs_encoded, "N T V E -> N T (V E)")
            goal = einops.rearrange(
                goal, "N T V E -> N T (V E)"
            )  # Assuming goal is already encoded or doesn't need it

            predicted_act, cbet_loss, loss_dict = cbet_model(obs_encoded, goal, act)

            # --- Combine Losses and Backpropagate ---
            loss = cbet_loss + aux_loss_weight * aux_loss
            loss_dict["aux_loss"] = aux_loss.item()  # Add aux loss to logs
            loss_dict["cbet_loss"] = cbet_loss.item()  # Add cbet loss to logs

            wandb.log({"train/{}".format(x): y for (x, y) in loss_dict.items()})
            encoder_param_snapshot = {
                name: p.clone().detach()
                for name, p in encoder.named_parameters()
                if p.requires_grad
            }

            loss.backward()

            # --- Gradient Check (Optional) ---
            num_elements_with_grad = sum(
                p.numel()
                for group in optimizer.param_groups
                for p in group["params"]
                if p.requires_grad and p.grad is not None and p.grad.abs().sum() != 0
            )

            print(
                f"Number of elements with nonzero grad after backward: {num_elements_with_grad}"
            )
            optimizer.step()
            num_changed = 0
            for name, p in encoder.named_parameters():
                if p.requires_grad:
                    before = encoder_param_snapshot[name]
                    after = p.detach()
                    if not torch.allclose(before, after):
                        print(f"Param {name} changed")
                        num_changed += 1

            print(f"{num_changed} encoder parameters changed after step")
            # print("num_parm_updataed:", updated_params)

        if epoch % cfg.save_every == 0:
            save_dir = Path(cfg.save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), save_dir / f"encoder_epoch_{epoch}.pth")
            torch.save(
                cbet_model.state_dict(), save_dir / f"cbet_model_epoch_{epoch}.pth"
            )
            print(f"Saved checkpoints to {save_dir}")

    avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
        cfg,
        num_evals=cfg.num_final_evals,
        num_eval_per_goal=cfg.num_final_eval_per_goal,
        videorecorder=video,
        epoch=cfg.epochs,
    )
    reward_history.append(avg_reward)
    if cfg.env.gym.id in ["pusht", "blockpush"]:
        metric_final = "final coverage" if cfg.env.gym.id == "pusht" else "entered"
        metric_max = "max coverage" if cfg.env.gym.id == "pusht" else "moved"
        metrics = {
            f"{metric_final} mean": sum(final_coverage) / len(final_coverage),
            f"{metric_final} max": max(final_coverage),
            f"{metric_final} min": min(final_coverage),
            f"{metric_max} mean": sum(max_coverage) / len(max_coverage),
            f"{metric_max} max": max(max_coverage),
            f"{metric_max} min": min(max_coverage),
        }
        wandb.log(metrics)
        metrics_history.append(metrics)

    with open("{}/completion_idx_final.json".format(save_path), "wb") as fp:
        pickle.dump(completion_id_list, fp)
    if cfg.env.gym.id == "pusht":
        final_eval_on_env = max([x["final coverage mean"] for x in metrics_history])
    elif cfg.env.gym.id == "blockpush":
        final_eval_on_env = max([x["entered mean"] for x in metrics_history])
    elif cfg.env.gym.id == "libero_goal":
        final_eval_on_env = max(reward_history)
    elif cfg.env.gym.id == "kitchen-v0":
        final_eval_on_env = avg_reward
    wandb.log({"final_eval_on_env": final_eval_on_env})
    return final_eval_on_env


if __name__ == "__main__":
    main()
