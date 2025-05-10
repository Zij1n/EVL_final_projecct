import os
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

# Load the normal predictor from DSINE hub repository
normal_predictor = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)

# Define input and output directories
input_dir = "/data/zijin/dynamo_repro_datasets/sim_kitchen_dataset/obses"
output_dir = (
    "/data/zijin/dynamo_repro_datasets/sim_kitchen_dataset/obses_surface_normal"
)
os.makedirs(output_dir, exist_ok=True)

# Get list of .pth files (assumes files end with .pth)
file_list = sorted([f for f in os.listdir(input_dir) if f.endswith(".pth")])

# For each file in the input directory
for file_name in tqdm(file_list, desc="Processing files"):
    input_path = os.path.join(input_dir, file_name)
    # Load the tensor with shape [189, 1, 3, 224, 224]
    data = torch.load(input_path)

    processed_normals = []
    # Process each of the 189 images
    for i in tqdm(range(data.shape[0]), desc=f"Processing {file_name}"):
        # Extract the RGB image (shape: [3, 224, 224])
        img_tensor = data[i, 0]  # shape [3, 224, 224]

        # Convert the tensor to a PIL image.
        # Note: ToPILImage expects the tensor to be in [0, 1] if float, or [0, 255] if uint8.
        pil_converter = ToPILImage()
        pil_image = pil_converter(img_tensor)

        # Resize to 800x800
        pil_image_800 = pil_image.resize((800, 800))

        # Get the normal map from the predictor (output shape: [3, 800, 800])
        res = normal_predictor.infer_pil(pil_image_800)[0]

        # Convert the range from [-1, 1] to [0, 1]
        normal = (res + 1) / 2

        # Resize the result from 800x800 to 224x224 using bilinear interpolation
        normal_resized = F.interpolate(
            normal.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)



        processed_normals.append(normal_resized)

    # Stack all processed normals; current shape: [189, 3, 224, 224]
    final_tensor = torch.stack(processed_normals, dim=0)
    # Add an extra dimension to match the target shape: [189, 1, 3, 224, 224]
    final_tensor = final_tensor.unsqueeze(1)

    # Save the processed tensor in the new folder with the same file name
    output_path = os.path.join(output_dir, file_name)
    torch.save(final_tensor, output_path)
