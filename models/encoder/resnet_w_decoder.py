import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

# Constants
COS_EPS = 1e-6  # Epsilon value for cosine similarity


# Define angular loss function
def angular_loss(norm_out, gt_norm, gt_norm_mask):
    """Calculate angular loss between predicted and ground truth normal vectors

    Args:
        norm_out:       (B, 4, ...) - predicted normals with kappa
        gt_norm:        (B, 3, ...) - ground truth normals
        gt_norm_mask:   (B, 1, ...) - mask for valid normal vectors

    Returns:
        loss: Mean angular error in radians
    """
    pred_norm = norm_out[:, 0:3, ...]  # Extract normal vectors (first 3 channels)
    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1).unsqueeze(1)
    valid_mask = torch.logical_and(gt_norm_mask, torch.abs(dot.detach()) < 1 - COS_EPS)
    angle = torch.acos(dot[valid_mask])
    return torch.mean(angle)


# Utility functions from DSINE submodules
def get_pixel_coords(h, w):
    # pixel array (1, 2, H, W)
    pixel_coords = np.ones((3, h, w)).astype(np.float32)
    x_range = np.concatenate([np.arange(w).reshape(1, w)] * h, axis=0)
    y_range = np.concatenate([np.arange(h).reshape(h, 1)] * w, axis=1)
    pixel_coords[0, :, :] = x_range + 0.5
    pixel_coords[1, :, :] = y_range + 0.5
    return torch.from_numpy(pixel_coords).unsqueeze(0)


def normal_activation(out, elu_kappa=True):
    normal, kappa = out[:, :3, :, :], out[:, 3:, :, :]
    normal = F.normalize(normal, p=2, dim=1)
    if elu_kappa:
        kappa = F.elu(kappa) + 1.0
    return torch.cat([normal, kappa], dim=1)


def upsample_via_bilinear(pred, mask, ratio):
    return F.interpolate(pred, scale_factor=ratio, mode="bilinear", align_corners=False)


def upsample_via_mask(pred, mask, ratio):
    B, C, H, W = pred.shape
    mask = mask.view(B, ratio * ratio, 9, H, W)
    mask = F.softmax(mask, dim=2)

    # Simplified implementation - full implementation would use the mask properly
    up_pred = F.interpolate(
        pred, scale_factor=ratio, mode="bilinear", align_corners=False
    )
    return up_pred


# Define UpSample modules
class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features, align_corners=True):
        super(UpSampleBN, self).__init__()
        self.skip_input = skip_input
        self.output_features = output_features
        self.align_corners = align_corners

        self.conv1 = nn.Conv2d(
            skip_input, output_features, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(output_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip_x):
        x = F.interpolate(
            x, size=skip_x.shape[2:], mode="bilinear", align_corners=self.align_corners
        )
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class UpSampleGN(nn.Module):
    def __init__(self, skip_input, output_features, align_corners=True):
        super(UpSampleGN, self).__init__()
        self.skip_input = skip_input
        self.output_features = output_features
        self.align_corners = align_corners

        self.conv1 = nn.Conv2d(
            skip_input, output_features, kernel_size=3, stride=1, padding=1
        )
        self.gn1 = nn.GroupNorm(32, output_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip_x):
        x = F.interpolate(
            x, size=skip_x.shape[2:], mode="bilinear", align_corners=self.align_corners
        )
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        return x


# Define the prediction head function
def get_prediction_head(in_channels, mid_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, 1),
    )


# Modified ResNet18 implementation to handle N,V,C,H,W inputs
class resnet18(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        output_dim: int = 512,  # fixed for resnet18; included for consistency with config
        unit_norm: bool = False,
        imgnet_norm: bool = True,  # False if dealing with surface noraml
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.pretrained = pretrained
        self.normalize = torchvision.transforms.Normalize(
            mean=mean,
            std=std,
        )
        self.unit_norm = unit_norm

        # Keep a reference to the original ResNet components for feature extraction
        self._resnet = resnet

        # Add the DSINE decoder
        self.dsine_decoder = DSINEDecoder(
            num_classes=4,  # 3 for normal + 1 for kappa
            NF=2048,
            BN=False,
            down=8,
            learned_upsampling=True,
        )

    def forward(self, x):
        # Save original shape to reshape back later
        orig_shape = x.shape

        # Flatten all dimensions before the last 3 (C,H,W)
        if len(orig_shape) > 4:
            # N,V,C,H,W -> (N*V),C,H,W
            x = x.reshape(-1, *orig_shape[-3:])

        x = self.normalize(x)
        out = self.resnet(x)
        out = self.flatten(out)

        if self.unit_norm:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)

        # Reshape back to include view dimension if original input had it
        if len(orig_shape) > 4:
            out = out.reshape(*orig_shape[:-3], -1)

        return out

    def extract_features(self, x):
        """Extract intermediate features from ResNet18 for use with DSINE decoder"""
        # Save original shape for reshaping back later
        orig_shape = x.shape

        # Flatten dimensions before the last 3 (C,H,W)
        if len(orig_shape) > 4:
            # N,V,C,H,W -> (N*V),C,H,W
            x = x.reshape(-1, *orig_shape[-3:])

        x_input = x.clone()  # Save input for feature list
        x = self.normalize(x)

        # Extract features using ResNet components
        conv1 = self._resnet.conv1
        bn1 = self._resnet.bn1
        relu = self._resnet.relu
        maxpool = self._resnet.maxpool
        layer1 = self._resnet.layer1
        layer2 = self._resnet.layer2
        layer3 = self._resnet.layer3
        layer4 = self._resnet.layer4

        # Extract features
        x0 = relu(bn1(conv1(x)))  # After conv1+bn1+relu
        x0_p = maxpool(x0)  # After maxpool

        x1 = layer1(x0_p)  # After layer1
        x2 = layer2(x1)  # After layer2
        x3 = layer3(x2)  # After layer3
        x4 = layer4(x3)  # After layer4

        # Return features in the format expected by DSINE decoder
        return [x_input, x0, x1, x2, x3, x4]

    def forward_with_dsine(self, x, intrins=None, mode="train"):
        """Forward pass using ResNet18 encoder + DSINE decoder"""
        # Save original shape
        orig_shape = x.shape

        # Check if we have N,V,C,H,W format
        multi_view = len(orig_shape) > 4

        # Flatten dimensions if needed
        if multi_view:
            # N,V,C,H,W -> (N*V),C,H,W
            batch_size, views = orig_shape[:2]
            x = x.reshape(-1, *orig_shape[-3:])

            # Also reshape intrinsics if provided
            if intrins is not None and intrins.dim() > 3:
                intrins = intrins.reshape(-1, *intrins.shape[-2:])

        # Create default intrinsics if none provided
        if intrins is None:
            # Default intrinsics for a 640x480 image
            batch_size = x.shape[0]
            default_intrins = torch.tensor(
                [
                    [193.9897, 0.0000, 111.5000],
                    [0.0000, 193.9897, 111.5000],
                    [0.0000, 0.0000, 1.0000],
                ],
                device=x.device,
            ).float()

            intrins = default_intrins.unsqueeze(0).repeat(batch_size, 1, 1)

        # Extract features and get decoder output
        features = self.extract_features(x)
        output = self.dsine_decoder(features, intrins, mode)

        # Reshape output back to include view dimension if needed
        if multi_view:
            # List of tensors - reshape each one
            output = [out.reshape(batch_size, views, *out.shape[1:]) for out in output]

        return output

    def compute_loss(self, x, gt_norm, gt_norm_mask, intrins=None):
        """
        Compute the loss between model prediction and ground truth normals

        Args:
            x: Input image tensor of shape (B, [V], C, H, W)
            gt_norm: Ground truth normal vectors of shape (B, [V], 3, H, W)
            gt_norm_mask: Mask indicating valid normal vectors of shape (B, [V], 1, H, W)
            intrins: Camera intrinsics, optional, of shape (B, [V], 3, 3)

        Returns:
            loss: Angular loss between predicted and ground truth normals
        """
        # Check if we have multi-view format
        multi_view = len(x.shape) > 4

        if multi_view:
            # For multi-view, we need to flatten before processing
            batch_size, views = x.shape[:2]
            x_flat = x.reshape(-1, *x.shape[-3:])
            gt_norm_flat = gt_norm.reshape(-1, *gt_norm.shape[-3:])
            gt_norm_mask_flat = gt_norm_mask.reshape(-1, *gt_norm_mask.shape[-3:])

            # Also flatten intrinsics if provided
            if intrins is not None and intrins.dim() > 3:
                intrins_flat = intrins.reshape(-1, *intrins.shape[-2:])
            else:
                intrins_flat = intrins

            # Forward pass
            out = self.forward_with_dsine(x_flat, intrins_flat)

            # Extract normals from output (first element of list)
            pred_out = out[0]

            # Calculate the angular loss
            return angular_loss(pred_out, gt_norm_flat, gt_norm_mask_flat)
        else:
            # For single view, use the original code
            out = self.forward_with_dsine(x, intrins)
            pred_out = out[0]
            return angular_loss(pred_out, gt_norm, gt_norm_mask)


# DSINE Decoder implementation (unchanged)
class DSINEDecoder(nn.Module):
    def __init__(
        self, num_classes=4, B=5, NF=2048, BN=False, down=8, learned_upsampling=True
    ):
        super(DSINEDecoder, self).__init__()

        # Define input channels for ResNet18 feature maps
        # ResNet18 channel dimensions: [3, 64, 64, 128, 256, 512]
        input_channels = [3, 64, 64, 128, 256, 512]

        # Use BN or GN
        UpSample = UpSampleBN if BN else UpSampleGN

        features = NF
        self.conv2 = nn.Conv2d(
            input_channels[5] + 2, features, kernel_size=1, stride=1, padding=0
        )
        self.up1 = UpSample(
            skip_input=features // 1 + input_channels[4] + 2,
            output_features=features // 2,
            align_corners=False,
        )
        self.up2 = UpSample(
            skip_input=features // 2 + input_channels[3] + 2,
            output_features=features // 4,
            align_corners=False,
        )

        if down == 8:
            i_dim = features // 4
        elif down == 4:
            self.up3 = UpSample(
                skip_input=features // 4 + input_channels[2] + 2,
                output_features=features // 8,
                align_corners=False,
            )
            i_dim = features // 8
        elif down == 2:
            self.up3 = UpSample(
                skip_input=features // 4 + input_channels[2] + 2,
                output_features=features // 8,
                align_corners=False,
            )
            self.up4 = UpSample(
                skip_input=features // 8 + input_channels[1] + 2,
                output_features=features // 16,
                align_corners=False,
            )
            i_dim = features // 16
        else:
            raise Exception("invalid downsampling ratio")

        self.downsample_ratio = down
        self.output_dim = num_classes

        self.pred_head = get_prediction_head(i_dim + 2, 128, num_classes)
        if learned_upsampling:
            self.mask_head = get_prediction_head(
                i_dim + 2, 128, 9 * self.downsample_ratio * self.downsample_ratio
            )
            self.upsample_fn = upsample_via_mask
        else:
            self.mask_head = lambda a: None
            self.upsample_fn = upsample_via_bilinear

        # pixel coordinates (1, 2, H, W)
        self.register_buffer("pixel_coords", get_pixel_coords(h=2000, w=2000))

    def ray_embedding(self, x, intrins, orig_H, orig_W):
        B, _, H, W = x.shape
        fu = intrins[:, 0, 0].unsqueeze(-1).unsqueeze(-1) * (W / orig_W)
        cu = intrins[:, 0, 2].unsqueeze(-1).unsqueeze(-1) * (W / orig_W)
        fv = intrins[:, 1, 1].unsqueeze(-1).unsqueeze(-1) * (H / orig_H)
        cv = intrins[:, 1, 2].unsqueeze(-1).unsqueeze(-1) * (H / orig_H)

        # (B, 2, H, W)
        uv = self.pixel_coords[:, :2, :H, :W].repeat(B, 1, 1, 1)
        uv[:, 0, :, :] = (uv[:, 0, :, :] - cu) / fu
        uv[:, 1, :, :] = (uv[:, 1, :, :] - cv) / fv
        return torch.cat([x, uv], dim=1)

    def forward(self, features, intrins, mode="train"):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[1],
            features[2],
            features[3],
            features[4],
            features[5],
        )
        _, _, orig_H, orig_W = features[0].shape

        # STEP 1: make top-left pixel (0.5, 0.5)
        intrins[:, 0, 2] += 0.5
        intrins[:, 1, 2] += 0.5

        x_d0 = self.conv2(self.ray_embedding(x_block4, intrins, orig_H, orig_W))
        x_d1 = self.up1(x_d0, self.ray_embedding(x_block3, intrins, orig_H, orig_W))

        if self.downsample_ratio == 8:
            x_feat = self.up2(
                x_d1, self.ray_embedding(x_block2, intrins, orig_H, orig_W)
            )
        elif self.downsample_ratio == 4:
            x_d2 = self.up2(x_d1, self.ray_embedding(x_block2, intrins, orig_H, orig_W))
            x_feat = self.up3(
                x_d2, self.ray_embedding(x_block1, intrins, orig_H, orig_W)
            )
        elif self.downsample_ratio == 2:
            x_d2 = self.up2(x_d1, self.ray_embedding(x_block2, intrins, orig_H, orig_W))
            x_d3 = self.up3(x_d2, self.ray_embedding(x_block1, intrins, orig_H, orig_W))
            x_feat = self.up4(
                x_d3, self.ray_embedding(x_block0, intrins, orig_H, orig_W)
            )

        out = self.pred_head(self.ray_embedding(x_feat, intrins, orig_H, orig_W))
        out = normal_activation(out, elu_kappa=True)

        mask = self.mask_head(self.ray_embedding(x_feat, intrins, orig_H, orig_W))
        up_out = self.upsample_fn(out, mask, self.downsample_ratio)
        up_out = normal_activation(up_out, elu_kappa=False)
        return [up_out]


# Updated example usage with multi-view input
def test_model():
    # Create the model
    model = resnet18(pretrained=True)

    # Create dummy data with shape [N, V, C, H, W]
    batch_size = 4
    views = 5
    channels = 3
    height = 224
    width = 224

    # Create dummy input image
    input_image = torch.randn(batch_size, views, channels, height, width)

    # Create dummy ground truth normals
    ground_truth_normals = torch.randn(batch_size, views, 3, height, width)
    ground_truth_normals = F.normalize(ground_truth_normals, p=2, dim=2)

    # Create dummy ground truth mask
    ground_truth_mask = torch.ones(batch_size, views, 1, height, width)

    # Create dummy intrinsics for each view
    intrinsics = (
        torch.tensor(
            [
                [193.9897, 0.0000, 111.5000],
                [0.0000, 193.9897, 111.5000],
                [0.0000, 0.0000, 1.0000],
            ]
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size, views, 1, 1)
    )  # Shape: [N, V, 3, 3]

    # Forward pass to check output shape
    output = model.forward_with_dsine(input_image, intrinsics)
    print(f"Output shape: {output[0].shape}")  # Should be [N, V, 4, 224, 224]

    # Check loss
    loss = model.compute_loss(
        input_image, ground_truth_normals, ground_truth_mask, intrinsics
    )
    print(f"Loss: {loss.item()}")  # Should be a scalar
