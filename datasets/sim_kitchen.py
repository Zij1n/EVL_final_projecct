import utils
import torch
import numpy as np
from pathlib import Path
from datasets.core import TrajectoryDataset
import time
from torch.utils.data import Dataset
import os
from einops import rearrange
from pathlib import Path
import torch, numpy as np, utils


class SimKitchenTrajectoryDataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory,
        prefetch: bool = True,
        onehot_goals: bool = False,
        obs_type: str = "both",  # Options: "obs_only", "normal_only", "both"
        new_view: bool = False,
        subset_size: int = None,  # Add subset_size parameter
    ):
        self.new_view = (
            new_view  # for both, new view will concat on v dim, otherwise c dim
        )

        self.data_directory = Path(data_directory)
        # load metadata
        states = torch.from_numpy(np.load(self.data_directory / "observations_seq.npy"))
        actions = torch.from_numpy(np.load(self.data_directory / "actions_seq.npy"))
        goals = torch.load(self.data_directory / "onehot_goals.pth")
        self.states, self.actions, self.goals = utils.transpose_batch_timestep(
            states, actions, goals
        )
        self.Ts = (
            np.load(self.data_directory / "existence_mask.npy")
            .sum(axis=0)
            .astype(int)
            .tolist()
        )

        # --- Subset Loading Logic ---
        if subset_size is not None and subset_size > 0 and subset_size < len(self.Ts):
            print(f"Loading subset of size: {subset_size}")
            self.states = self.states[:subset_size]
            self.actions = self.actions[:subset_size]
            self.goals = self.goals[:subset_size]
            self.Ts = self.Ts[:subset_size]
        # --- End Subset Loading Logic ---

        self.prefetch = prefetch
        assert obs_type in ["obs_only", "normal_only", "both"], (
            f"Invalid obs_type: {obs_type}. Must be 'obs_only', 'normal_only', or 'both'."
        )
        print("obs type:", obs_type)
        self.obs_type = obs_type
        self.onehot_goals = onehot_goals

        if self.prefetch:
            N = len(self.Ts)
            # debug!

            # build frame‐offsets
            offsets = [0] * N
            for i in range(1, N):
                offsets[i] = offsets[i - 1] + self.Ts[i - 1]
            self.offsets = offsets
            total_frames = offsets[-1] + self.Ts[-1]

            # inspect one file for dims
            # Determine channel count based on obs_type
            # We assume RGB and Normal have the same base channel count (C=3)
            sample_path = (
                self.data_directory
                / ("obses" if obs_type != "normal_only" else "obses_surface_normal")
                / "000.pth"
            )
            sample_data = torch.load(
                sample_path, map_location="cpu"
            )  # [T₀, V, C, H, W]
            _, V, C, H, W = sample_data.shape

            if self.obs_type == "both":
                C_total = C * 2
            elif self.obs_type == "obs_only" or self.obs_type == "normal_only":
                C_total = C
            else:  # Should not happen due to assert
                raise ValueError(f"Invalid obs_type: {self.obs_type}")

            # preallocate float32 buffer
            self.obses = torch.zeros(
                (total_frames, V, C_total, H, W), dtype=torch.float32
            )

            for i in range(N):
                T_i = self.Ts[i]
                start = offsets[i]
                end = start + T_i

                if self.obs_type == "obs_only":
                    obs_i = (
                        torch.load(
                            self.data_directory / "obses" / f"{i:03d}.pth",
                            map_location="cpu",
                        ).to(torch.float32)
                        / 255.0
                    )  # [T_i, V, C, H, W]
                    self.obses[start:end] = obs_i
                elif self.obs_type == "normal_only":
                    norm_i = torch.load(
                        self.data_directory / "obses_surface_normal" / f"{i:03d}.pth",
                        map_location="cpu",
                    ).to(torch.float32)  # [T_i, V, C, H, W]
                    self.obses[start:end] = norm_i
                elif self.obs_type == "both":
                    obs_i = (
                        torch.load(
                            self.data_directory / "obses" / f"{i:03d}.pth",
                            map_location="cpu",
                        ).to(torch.float32)
                        / 255.0
                    )
                    norm_i = torch.load(
                        self.data_directory / "obses_surface_normal" / f"{i:03d}.pth",
                        map_location="cpu",
                    ).to(torch.float32)
                    # Fill the preallocated buffer
                    self.obses[start:end, :, :C, :, :] = obs_i
                    self.obses[start:end, :, C:, :, :] = norm_i
                # No else needed due to assert in __init__

    def get_frames(self, idx, frames):
        # t0 = time.perf_counter()
        if self.prefetch:
            # t1 = time.perf_counter()
            start = self.offsets[idx]
            # indices = [start + f for f in frames]
            # L = len(frames)
            # t2 = time.perf_counter()
            # print("prefetch index calc:", t2 - t1)
            assert frames[-1] - frames[0] + 1 == len(frames), "frames not continuous"
            obs = self.obses[start + frames[0] : start + frames[-1] + 1]
            # if i do obs = self.obses[indices] it is 200x slower?????
            # t3 = time.perf_counter()
            # print("prefetch obs access:", t3 - t2)
        else:
            # Load data based on obs_type when not prefetching
            if self.obs_type == "obs_only":
                obs = (
                    torch.load(
                        self.data_directory / "obses" / f"{idx:03d}.pth",
                        map_location="cpu",
                    ).to(torch.float32)[frames]
                    / 255.0
                )
            elif self.obs_type == "normal_only":
                obs = torch.load(
                    self.data_directory / "obses_surface_normal" / f"{idx:03d}.pth",
                    map_location="cpu",
                ).to(torch.float32)[frames]
            elif self.obs_type == "both":
                obs_rgb = (
                    torch.load(
                        self.data_directory / "obses" / f"{idx:03d}.pth",
                        map_location="cpu",
                    ).to(torch.float32)[frames]
                    / 255.0
                )
                norm = torch.load(
                    self.data_directory / "obses_surface_normal" / f"{idx:03d}.pth",
                    map_location="cpu",
                ).to(torch.float32)[frames]
                obs = torch.cat([obs_rgb, norm], dim=2)  # Concatenate along channel dim
            # No else needed due to assert in __init__

        # t6 = time.perf_counter()
        act = self.actions[idx, frames]
        # t7 = time.perf_counter()
        # print("load action:", t7 - t6)

        mask = torch.ones(len(frames), dtype=torch.float32)
        # t8 = time.perf_counter()
        # print("make mask:", t8 - t7)

        if self.new_view == True:
            obs = rearrange(
                obs,
                "N  V (two C) H W -> N ( V two)  C H W",
                two=2,
                C=3,
            )

        if self.onehot_goals:
            # t9 = time.perf_counter()
            goal = self.goals[idx, frames]
            # t10 = time.perf_counter()
            # print("load goal:", t10 - t9)
            return obs, act, mask, goal

        return obs, act, mask

    def get_seq_length(self, idx):
        return self.Ts[idx]

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.Ts)):
            T = self.Ts[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.Ts)

    def __getitem__(self, idx):
        T = self.Ts[idx]
        return self.get_frames(idx, range(T))


class SurfaceNormalDataset(Dataset):
    # Dataset for BYOL training
    def __init__(self, data_directory, partial=None):
        self.data_directory = Path(data_directory)
        self.surface_normal_dir = self.data_directory / "obses_surface_normal"
        self.partial = partial  # <<< ADD partial

        # Load existence mask
        existence_mask = np.load(self.data_directory / "existence_mask.npy")  # [T, N]
        self.Ts = existence_mask.sum(axis=0).astype(int).tolist()

        # Find and sort all .pth files
        self.pth_files = sorted(
            [f for f in os.listdir(self.surface_normal_dir) if f.endswith(".pth")]
        )

        if self.partial > 0:
            self.pth_files = self.pth_files[: self.partial]
            self.Ts = self.Ts[: self.partial]

        assert len(self.pth_files) == len(self.Ts), (
            f"Mismatch: {len(self.pth_files)} files vs {len(self.Ts)} sequences from existence_mask."
        )

        # Build offsets
        self.offsets = [0] * len(self.Ts)
        for i in range(1, len(self.Ts)):
            self.offsets[i] = self.offsets[i - 1] + self.Ts[i - 1]

        total_frames = self.offsets[-1] + self.Ts[-1]
        self.N = total_frames

        # Inspect one file to get C, H, W
        sample = torch.load(
            self.surface_normal_dir / self.pth_files[0], map_location="cpu"
        )
        _, _, C, H, W = sample.shape

        # Preallocate
        self.obses = torch.zeros((self.N, C, H, W), dtype=torch.float32)

        # Fill
        idx = 0
        for file in self.pth_files:
            data = torch.load(
                self.surface_normal_dir / file, map_location="cpu"
            )  # [N, V, 3, 224, 224]
            n, v, c, h, w = data.shape
            data = data.view(-1, c, h, w)  # flatten (N*V, C, H, W)
            self.obses[idx : idx + data.shape[0]] = data
            idx += data.shape[0]

        assert idx == self.N, f"Mismatch: filled {idx}, expected {self.N}"
        self.obses = self.obses.share_memory_()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.obses[idx]
