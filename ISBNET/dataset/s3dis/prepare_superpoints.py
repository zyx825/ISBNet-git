import numpy as np
import torch

import glob


files = sorted(glob.glob("learned_superpoint_graph_segmentations/*.npy"))
print(len(files))

for file in files:
    chunks = file.split("/")[-1].split(".")
    area = chunks[0]
    room = chunks[1]

    spp = np.load(file, allow_pickle=True).item()["segments"]
    torch.save((spp), f"/home/hhroot/ISBNet/dataset/s3dis/superpoints/{area}_{room}.pth")
