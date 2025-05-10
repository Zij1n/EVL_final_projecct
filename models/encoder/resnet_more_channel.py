import torch
import torch.nn as nn
import torchvision


class resnet18(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        output_dim: int = 512,  # fixed for resnet18; included for consistency with config
        unit_norm: bool = False,
        input_channels: int = 6,  # Modified here
    ):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)

        # Modify the first convolution layer to accept 6 channels
        self.first_conv = nn.Conv2d(
            input_channels,
            resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=resnet.conv1.bias,
        )

        # If pretrained, copy weights for the first 3 channels from original conv1
        if pretrained:
            with torch.no_grad():
                self.first_conv.weight[:, :3, :, :] = resnet.conv1.weight
                # Initialize additional channels using default CNN initialization
                nn.init.kaiming_normal_(
                    self.first_conv.weight[:, 3:, :, :],
                    mode="fan_out",
                    nonlinearity="relu",
                )

        # Replace the first layer and keep the rest of ResNet
        self.resnet = nn.Sequential(self.first_conv, *list(resnet.children())[1:-1])

        self.flatten = nn.Flatten()
        self.pretrained = pretrained

        # Normalize only the original 3 channels; additional channels are not normalized
        self.normalize_rgb = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # Normalize the last 3 channels (surface normal)
        self.normalize_normal = torchvision.transforms.Normalize(
            mean=[0.4583, 0.5643, 0.8529], std=[0.1479, 0.3041, 0.07]
        )
        self.unit_norm = unit_norm

    def forward(self, x):
        dims = len(x.shape)
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            # flatten all dimensions to batch, then reshape back at the end
            x = x.reshape(-1, *orig_shape[-3:])

        # Normalize the first 3 channels (RGB) and the last 3 channels (surface normal)
        x_rgb_norm = self.normalize_rgb(x[:, :3, :, :])
        x_normal_norm = self.normalize_normal(x[:, 3:, :, :])
        x = torch.cat((x_rgb_norm, x_normal_norm), dim=1)

        out = self.resnet(x)
        out = self.flatten(out)
        if self.unit_norm:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
        if dims == 3:
            out = out.squeeze(0)
        elif dims > 4:
            out = out.reshape(*orig_shape[:-3], -1)
        return out
