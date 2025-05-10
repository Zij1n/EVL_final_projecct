import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import models
from byol_pytorch import BYOL
from datasets.sim_kitchen import SurfaceNormalDataset
from models.lars import LARS
from accelerate.utils import DistributedDataParallelKwargs
import random
from torchvision import transforms as T
from torch import nn
import torch.nn.functional as F


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


@hydra.main(
    config_path="configs/train_surface_enc_byol",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    # Setup Accelerate with native SyncBatchNorm + WandB tracking
    accelerator = Accelerator(
        log_with=["wandb"],
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    if accelerator.is_main_process:
        wandb_run = accelerator.get_tracker("wandb", unwrap=True)
    # Only main process initializes trackers
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb_cfg["save_path"] = str(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.project,
            config=wandb_cfg,
            init_kwargs={
                "wandb": {
                    "reinit": False,
                    "settings": {"start_method": "thread"},
                },
            },
        )

    # Important: get tracker AFTER init_trackers
    wandb_run = accelerator.get_tracker("wandb", unwrap=True)

    # --- Model backbone ---
    if cfg.model.name == "resnet18":
        backbone = models.resnet18(pretrained=cfg.model.pretrained)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")

    # Move backbone to correct device before BYOL init
    backbone = backbone.to(accelerator.device)

    do_normalize = cfg.training.do_normalize

    normalization = (
        T.Normalize(mean=[0.4583, 0.5643, 0.8529], std=[0.1479, 0.3041, 0.07])
        if do_normalize
        else torch.nn.Identity()
    )

    aug1 = torch.nn.Sequential(
        # RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
        # T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
        T.RandomResizedCrop((224, 224)),
        normalization,
    )

    aug2 = torch.nn.Sequential(
        # RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
        # T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
        T.RandomResizedCrop((224, 224)),
        normalization,
    )
    model = BYOL(
        backbone,
        image_size=cfg.model.image_size,
        hidden_layer=cfg.model.byol.hidden_layer,
        projection_size=cfg.model.byol.projection_size,
        projection_hidden_size=cfg.model.byol.projection_hidden_size,
        moving_average_decay=cfg.model.byol.moving_average_decay,
        augment_fn=aug1,
        augment_fn2=aug2,
    )

    # --- Dataset & Dataloader ---
    dataset = SurfaceNormalDataset(
        data_directory=cfg.data.dir,
        partial=cfg.data.partial,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=getattr(cfg.training, "shuffle", True),
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        drop_last=True,
    )

    # --- Optimizer ---
    if cfg.optimizer.name == "lars":
        optimizer = LARS(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
            trust_coefficient=cfg.optimizer.lars.trust_coefficient,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")

    # --- Scheduler ---
    steps_per_epoch = len(dataloader)
    total_steps = cfg.training.num_epochs * steps_per_epoch
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch

    scheduler_warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=0.0,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_steps],
    )

    # --- Prepare for distributed, AMP, and device placement ---
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    if accelerator.is_main_process:
        wandb_run.watch(model.online_encoder)
    global_step = 0
    for epoch in tqdm(range(cfg.training.num_epochs)):
        model.train()

        # Save model checkpoint every n epochs
        if (epoch + 1) % cfg.training.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(
                hydra.core.hydra_config.HydraConfig.get().run.dir,
                f"epoch_{epoch + 1}.pth",
            )
            accelerator.save(model.state_dict(), checkpoint_path)

        epoch_loss = 0.0

        for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
            # dataloader tensors are already on device
            images = batch[0] if isinstance(batch, (list, tuple)) else batch

            loss = model(images)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update momentum encoder
            if hasattr(model, "update_moving_average"):
                model.update_moving_average()

            global_step += 1
            epoch_loss += loss.item()

        # End-of-epoch logging
        # Calculate cosine similarity and std dev for the first batch
        first_batch = next(iter(dataloader))
        images = (
            first_batch[0] if isinstance(first_batch, (list, tuple)) else first_batch
        )

        # Get representations from the online encoder
        online_repr = model.online_encoder(images)[1]

        # Calculate cosine similarity
        online_repr_norm = F.normalize(online_repr, dim=1)
        cosine_sim_matrix = torch.matmul(online_repr_norm, online_repr_norm.T)
        # Exclude diagonal for average
        avg_cosine_similarity = (
            cosine_sim_matrix.sum() - torch.diag(cosine_sim_matrix).sum()
        ) / (online_repr_norm.size(0) * (online_repr_norm.size(0) - 1))

        # Calculate standard deviation
        std_dev = torch.std(online_repr, dim=0).mean()

        accelerator.log(
            {
                "epoch/loss": epoch_loss / steps_per_epoch,
                "epoch/lr": scheduler.get_last_lr()[0],
                "epoch/avg_cosine_similarity": avg_cosine_similarity.item(),
                "epoch/representation_std_dev": std_dev.item(),
            },
            step=global_step,
        )

    # Finalize wandb tracker
    accelerator.end_training()


if __name__ == "__main__":
    main()
