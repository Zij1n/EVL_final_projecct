import torch
import torch.nn as nn
import torchvision
import einops
from typing import List, Tuple


class MultiviewMLP(nn.Module):
    def __init__(
        self,
        encoders: List[nn.Module],
        normalizations: List[Tuple[List, List]],
        output_dim: int,
    ):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.normalizations = []
        for mean, std in normalizations:
            self.normalizations.append(
                torchvision.transforms.Normalize(mean=mean, std=std)
            )

        # Assume all encoders output same dim
        encoder_output_dim = None
        for encoder in encoders:
            if hasattr(encoder, "output_dim"):
                encoder_output_dim = encoder.output_dim
                break
        if encoder_output_dim is None:
            raise ValueError("Encoder output_dim must be accessible.")

        input_dim = len(encoders) * encoder_output_dim
        hidden_dim = input_dim * 4

        self.fuse_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        orig_shape = x.shape  # NTVCHW or TVCHW
        x = einops.rearrange(x, "... V C H W -> (...) V C H W")
        outputs = []
        for i, encoder in enumerate(self.encoders):
            this_view = x[:, i]
            # this_view = self.normalizations[i](this_view)  # Uncomment if needed
            outputs.append(encoder(this_view))
        out = torch.stack(outputs, dim=-1)  # (batch, feature_dim, V)
        out = out.flatten(start_dim=-2)  # (batch, feature_dim * V)
        out = self.fuse_mlp(out)
        return out
