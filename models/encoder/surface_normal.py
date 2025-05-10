# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from PIL import Image


# class DSINE(nn.Module):
#     def __init__(self):
#         super(DSINE, self).__init__()
#         # Load the DSINE normal predictor from the hub repository.
#         self.normal_predictor = torch.hub.load(
#             "hugoycj/DSINE-hub", "DSINE", trust_repo=True
#         )
#         self.normal_predictor

#     def forward(self, pil_image):
#         # Ensure the input is a PIL Image.
#         if not isinstance(pil_image, Image.Image):
#             raise TypeError("Input must be a PIL Image")

#         # Resize the PIL image to 800x800.
#         pil_image_800 = pil_image.resize((800, 800))

#         # Get the normal map from the predictor.
#         # The output is expected to be a tensor of shape [3, 800, 800].
#         normal = self.normal_predictor.infer_pil(pil_image_800)[0]

#         # Convert the normal map's range from [-1, 1] to [0, 1].
#         normal = (normal + 1) / 2

#         # Resize the normal map from 800x800 to 224x224 using bilinear interpolation.
#         normal_resized = F.interpolate(
#             normal.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
#         ).squeeze(0)

#         return normal_resized


# if __name__ == "__main__":
#     from torchvision.transforms import ToPILImage

#     # Path to the test .pth file that contains the tensor
#     test_pth_path = "/data/zijin/dynamo_repro_datasets/sim_kitchen_dataset/obses_surface_normal/000.pth"  # Replace with your actual file path

#     # Load the tensor: expected shape [N, 1, 3, 224, 224]
#     try:
#         data = torch.load(test_pth_path)
#         img_tensor = data[0, 0]  # Get the first sample: shape [3, 224, 224]
#         print(f"Loaded tensor from {test_pth_path} with shape {img_tensor.shape}")
#     except Exception as e:
#         print(f"Error loading '{test_pth_path}': {e}")
#         # If loading fails, create a dummy tensor
#         img_tensor = torch.ones((3, 224, 224), dtype=torch.float32)
#         print("Created a dummy tensor as fallback.")

#     # Convert tensor to PIL image (tensor should be in [0, 1] range for float)
#     pil_converter = ToPILImage()
#     pil_img = pil_converter(img_tensor)

#     # Create an instance of the DSINENormalResizer
#     model = DSINE()

#     # Process the image without gradients
#     with torch.no_grad():
#         normal_resized = model(pil_img)

#     # Convert the result to a PIL image for saving
#     normal_np = normal_resized.mul(255).byte().permute(1, 2, 0).cpu().numpy()
#     normal_img = Image.fromarray(normal_np)

#     # Save the output image
#     output_path = "test_res.png"
#     normal_img.save(output_path)
#     print(f"Saved processed normal image to {output_path}")
# # import os
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from PIL import Image


# # class DSINE(nn.Module):
# #     def __init__(self):
# #         super(DSINE, self).__init__()
# #         # Load the DSINE normal predictor from the hub repository
# #         self.normal_predictor = torch.hub.load(
# #             "hugoycj/DSINE-hub", "DSINE", trust_repo=True
# #         )
# #         self.normal_predictor

# #     def forward(self, img_tensor):
# #         # Ensure it's a tensor with last 3 dims [3, H, W]
# #         if not isinstance(img_tensor, torch.Tensor):
# #             raise TypeError("Input must be a torch.Tensor")
# #         if img_tensor.shape[-3] != 3:
# #             raise ValueError("The last dimension-3 block must be [3, H, W]")

# #         # Flatten leading dimensions if present
# #         original_shape = img_tensor.shape[:-3]  # batch or multi-batch dims
# #         flat_tensor = img_tensor.view(
# #             -1, 3, img_tensor.shape[-2], img_tensor.shape[-1]
# #         )  # [N, 3, H, W]

# #         results = []
# #         for img in flat_tensor:
# #             # Resize to [3, 800, 800]
# #             img_resized = F.interpolate(
# #                 img.unsqueeze(0), size=(800, 800), mode="bilinear", align_corners=False
# #             )
# #             print(img_resized.shape)
# #             # Predict normal map
# #             normal = self.normal_predictor.infer_tensor(img_resized)[0]

# #             # Convert from [-1, 1] to [0, 1]
# #             normal = (normal + 1) / 2

# #             # Resize to [3, 224, 224]
# #             normal_resized = F.interpolate(
# #                 normal.unsqueeze(0),
# #                 size=(224, 224),
# #                 mode="bilinear",
# #                 align_corners=False,
# #             ).squeeze(0)

# #             results.append(normal_resized)

# #         # Stack and reshape back to original leading dims + [3, 224, 224]
# #         result_tensor = torch.stack(results, dim=0)
# #         result_tensor = result_tensor.view(*original_shape, 3, 224, 224)

# #         return result_tensor


# # if __name__ == "__main__":
# #     # Example: load tensor with shape [N, 1, 3, 224, 224]
# #     test_pth_path = "/data/zijin/dynamo_repro_datasets/sim_kitchen_dataset/obses_surface_normal/000.pth"
# #     data = torch.load(test_pth_path)
# #     img_tensor = data[0, 0]  # Get the first sample: shape [3, 224, 224]

# #     model = DSINE()

# #     with torch.no_grad():
# #         normals = model(img_tensor)

# #     # Save the first result of the flattened structure as an image for testing
# #     first_result = normals.view(-1, 3, 224, 224)[0]
# #     normal_np = first_result.mul(255).byte().permute(1, 2, 0).cpu().numpy()
# #     normal_img = Image.fromarray(normal_np)
# #     normal_img.save("test_res.png")
# #     print("Saved first processed normal map to test_res.png")
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToPILImage  # For tensor to PIL conversion


class DSINE(nn.Module):
    def __init__(self):
        super(DSINE, self).__init__()
        # Load the DSINE normal predictor from the hub repository.
        self.normal_predictor = torch.hub.load(
            "hugoycj/DSINE-hub", "DSINE", trust_repo=True
        )
        self.pil_converter = ToPILImage()  # For converting tensors to PIL images

    def _process_single(self, image):
        """
        Process a single image (PIL Image) with the DSINE pipeline.
        """
        # Resize the PIL image to 800x800.
        pil_image_800 = image.resize((800, 800))
        # Get the normal map from the predictor.
        # The output is expected to be a tensor of shape [3, 800, 800].
        normal = self.normal_predictor.infer_pil(pil_image_800)[0]
        # Convert the normal map's range from [-1, 1] to [0, 1].
        normal = (normal + 1) / 2
        # Resize the normal map from 800x800 to 224x224 using bilinear interpolation.
        normal_resized = F.interpolate(
            normal.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)
        return normal_resized

    def forward(self, image_input):
        # If the input is a tensor, handle both single image and batched images.
        if isinstance(image_input, torch.Tensor):
            # If the tensor has less than 3 dims, it's not valid.
            if image_input.ndim < 3:
                raise ValueError(
                    "Tensor input must have at least 3 dimensions: [C, H, W] or [B, C, H, W], etc."
                )
            # Determine the shape and flatten leading dims if necessary.
            *batch_dims, C, H, W = image_input.shape
            # Flatten all leading dimensions (if any) into a single batch dimension.
            flat_batch = image_input.reshape(-1, C, H, W)
            outputs = []
            for img in flat_batch:
                # If the tensor has 1 channel, replicate to 3 channels.
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                # If the tensor has more than 3 channels, take the first 3.
                elif img.shape[0] > 3:
                    img = img[:3, ...]
                # Convert the tensor to a PIL image.
                pil_img = self.pil_converter(img)
                # Process the image.
                out = self._process_single(pil_img)
                outputs.append(out)
            # Stack outputs back into a batch.
            batched_output = torch.stack(outputs, dim=0)
            # If there were leading dimensions, reshape back to original batch dimensions.
            if batch_dims:
                batched_output = batched_output.reshape(*batch_dims, 3, 224, 224)
            return batched_output

        # If the input is a PIL image, process it directly.
        elif isinstance(image_input, Image.Image):
            return self._process_single(image_input)
        else:
            raise TypeError(
                "Input must be a PIL Image or a torch.Tensor with shape [C, H, W] or [B, C, H, W], etc."
            )


if __name__ == "__main__":
    # Path to the test .pth file that contains the tensor.
    test_pth_path = "/data/zijin/dynamo_repro_datasets/sim_kitchen_dataset/obses_surface_normal/000.pth"  # Replace with your actual file path

    # Load the tensor: expected shape [N, 1, 3, 224, 224]
    try:
        data = torch.load(test_pth_path)
        # For demonstration, extract the first batch item: shape [3, 224, 224]
        img_tensor = data[0]
        print(f"Loaded tensor from {test_pth_path} with shape {img_tensor.shape}")
    except Exception as e:
        print(f"Error loading '{test_pth_path}': {e}")
        # If loading fails, create a dummy tensor of shape [3, 224, 224]
        img_tensor = torch.ones((3, 224, 224), dtype=torch.float32)
        print("Created a dummy tensor as fallback.")

    # To simulate a batch, we can add a batch dimension.
    batch_tensor = img_tensor.unsqueeze(0)  # Now shape is [1, 3, 224, 224]

    # Create an instance of DSINE.
    model = DSINE()

    # Process the image(s) without gradients.
    with torch.no_grad():
        normal_resized = model(batch_tensor)

    # If the output is batched, convert the first image to a PIL image for saving.
    if normal_resized.ndim == 4:
        output_tensor = normal_resized[0]
    else:
        output_tensor = normal_resized

    # Convert the result to a PIL image for saving.
    print(output_tensor.shape)
    normal_np = output_tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    normal_img = Image.fromarray(normal_np)

    # Save the output image.
    output_path = "test_res.png"
    normal_img.save(output_path)
    print(f"Saved processed normal image to {output_path}")
