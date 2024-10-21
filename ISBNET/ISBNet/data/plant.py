import numpy as np
import torch

from .custom import CustomDataset

class PLANTDataset(CustomDataset):

    CLASSES = (
        "stem",
        "leaf",
    )
    BENCHMARK_SEMANTIC_IDXS = [i for i in range(20)]  # NOTE DUMMY values just for save results
    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        ret = super().getInstanceInfo(xyz, instance_label, semantic_label)
        instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
        # (Optional) Reorder class id if needed.
        instance_cls = [x  if x != -100 else x for x in instance_cls]
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    def load(self, filename):
        if self.prefix == "tea_for_test_pth":
            xyz, rgb = torch.load(filename)
            semantic_label = np.zeros(xyz.shape[0], dtype=np.int64)
            instance_label = np.zeros(xyz.shape[0], dtype=np.int64)
        else:
            xyz, rgb, semantic_label, instance_label = torch.load(filename)

        # NOTE currently the plant dataset does not have semantic per point, we will add later
        # semantic_label = np.zeros(xyz.shape[0], dtype=np.long)

        # (Optional) Calculate instance label if needed.
        # instance_label = self.calculate_instance_label(xyz)

        spp = np.arange(xyz.shape[0], dtype=np.int64)

        return xyz, rgb, semantic_label, instance_label, spp
