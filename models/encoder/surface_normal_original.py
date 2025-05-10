from DSINE.projects.dsine.test_minimal import get_surface_normal
import sys
import torch
if __name__ == "__main__":
    test_pth_path = "/home/zifan/baby-to-robot/dynamo_repro_datasets/sim_kitchen_dataset/obses/000.pth"
    data = torch.load(test_pth_path)
    img_tensor_list = []
    for i in range(data.shape[0]):
        img_tensor = data[i, 0]
        img_tensor_list.append(img_tensor)
    get_surface_normal(img_tensor_list)