# https://github.com/meidachen/STPLS3D/blob/main/HAIS/data/prepare_data_inst_instance_stpls3d.py
import numpy as np
import pandas as pd
import torch

import glob
import json
import math
import os
import random


# 读取点云进行分割，该函数通过将点云数据划分为大小相等的块，实现了对点云数据的分割和组织。
def splitPointCloud(cloud, size=50.0, stride=50):
    # 得到点云数据的边界信息，即x，y，z的最大值
    limitMax = np.amax(cloud[:, 0:3], axis=0)
    # 获取宽和深的块数目
    width = int(np.ceil((limitMax[0] - size) / stride)) + 1
    depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
    # 存储了所有块的坐标
    cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
    blocks = []
    for (x, y) in cells:
        xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
        ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
        # 获取到在块内的点
        cond = xcond & ycond
        block = cloud[cond, :]
        blocks.append(block)
    return blocks


# 该函数根据文件路径列表中文件的命名规则，提取文件名中的数字要素，并根据这些数字要素判断是否将该文件路径添加到结果列表中。
def getFiles(files, fileSplit):
    res = []
    for filePath in files:
        name = os.path.basename(filePath)
        if name[:3].isdigit():
            num = name[:3]
        elif name[:2].isdigit():
            num = name[:2]
        else:
            num = name[:1]
        # num = name[:3] if name[:3].isdigit() else name[:1]
        if int(num) in fileSplit:
            res.append(filePath)
    return res


def dataAug(file, semanticKeep):
    # 将点云文件转为数组形式
    points = pd.read_csv(file, header=None).values
    angle = random.randint(1, 359)
    # 将角度转换为弧度
    angleRadians = math.radians(angle)
    # 旋转矩阵 用来将点云数据旋转
    rotationMatrix = np.array(
        [
            [math.cos(angleRadians), -math.sin(angleRadians), 0],
            [math.sin(angleRadians), math.cos(angleRadians), 0],
            [0, 0, 1],
        ]
    )
    # 将点云数据的前三列进行旋转
    points[:, :3] = points[:, :3].dot(rotationMatrix)
    # 判断points中的第七列是否在semanticKeep中，将在其中的返回到pointsKept 中
    pointsKept = points[np.in1d(points[:, 6], semanticKeep)]
    return pointsKept


# 用于准备点云文件的pth文件
# files（文件列表）、split（数据集的划分，例如"train"、"val"或"test1"）、
# outPutFolder（输出文件夹路径）、AugTimes（数据增强的次数，默认为0）和 crop_size（裁剪块的大小，默认为50）。
def preparePthFiles(files, split, outPutFolder, AugTimes=0, crop_size=50):
    # save the coordinates so that we can merge the data to a single scene保存坐标，以便将数据合并到一个场景中
    # after segmentation for visualization
    outJsonPath = os.path.join(outPutFolder, "coordShift.json")
    # coordShift 是一个字典，用于存储每个场景的坐标偏移。
    coordShift = {}
    # used to increase z range if it is smaller than this,，zThreshold = 6 是一个阈值
    # over come the issue where spconv may crash for voxlization.
    # 设置一个阈值为6，用来限制z轴坐标的太小的情况
    zThreshold = 6

    # 将相关类映射为{1,...,14}，忽略的类映射为-100
    remapper = np.ones(150) * (-100)
    for i, x in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
        remapper[x] = i
    # Map instance to -100 based on selected semantic
    # (change a semantic to -100 if you want to ignore it for instance)
    remapper_disableInstanceBySemantic = np.ones(150) * (-100)
    for i, x in enumerate([-100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
        remapper_disableInstanceBySemantic[x] = i

    # only augment data for these classes。表示要数据增强的类别
    semanticKeep = [0, 2, 3, 7, 8, 9, 12, 13]

    counter = 0
    # 对于在files中的file进行增强
    for file in files:

        for AugTime in range(AugTimes + 1):
            if AugTime == 0:
                points = pd.read_csv(file, header=None).values
            else:
                points = dataAug(file, semanticKeep)
            name = os.path.basename(file).strip(".txt") + "_%d" % AugTime  # 名称加上AugTime
            # split表示是train还是test还是val
            if split != "test1":
                # split 不是 "test1"，则将点云数据的全局偏移量存储在 coordShift["globalShift"] 中。
                coordShift["globalShift"] = list(points[:, :3].min(0))
            points[:, :3] = points[:, :3] - points[:, :3].min(0)
            # 使用 splitPointCloud 函数将点云数据划分为大小为 crop_size 的块，结果存储在 blocks 列表中。
            blocks = splitPointCloud(points, size=crop_size, stride=crop_size)
            for blockNum, block in enumerate(blocks):
                # 如果块中的点数大于10000，则根据指定的阈值 zThreshold，在块数据的末尾添加一个点，以增加z轴范围。同时打印一条消息指示范围z小于阈值。
                if len(block) > 10000:
                    outFilePath = os.path.join(outPutFolder, name + str(blockNum) + "_inst_nostuff.pth")
                    if block[:, 2].max(0) - block[:, 2].min(0) < zThreshold:   # 如果z坐标范围过小，则在后面添加一个点,维度为0
                        block = np.append(
                            block,
                            [
                                [
                                    block[:, 0].mean(0),
                                    block[:, 1].mean(0),
                                    block[:, 2].max(0) + (zThreshold - (block[:, 2].max(0) - block[:, 2].min(0))),
                                    block[:, 3].mean(0),
                                    block[:, 4].mean(0),
                                    block[:, 5].mean(0),
                                    -100,
                                    -100,
                                ]
                            ],
                            axis=0,
                        )
                        print("range z is smaller than threshold ")
                        print(name + str(blockNum) + "_inst_nostuff")
                    if split != "test1":
                        outFileName = name + str(blockNum) + "_inst_nostuff"
                        coordShift[outFileName] = list(block[:, :3].mean(0))
                    # 是将中心化后的坐标数据转换为连续内存布局的Numpy数组。这样做可以提高数据在计算和存储时的效率。
                    coords = np.ascontiguousarray(block[:, :3] - block[:, :3].mean(0))

                    # coords = block[:, :3]
                    colors = np.ascontiguousarray(block[:, 3:6]) / 127.5 - 1

                    coords = np.float32(coords)
                    colors = np.float32(colors)
                    if split != "test1":
                        sem_labels = np.ascontiguousarray(block[:, -2])
                        sem_labels = sem_labels.astype(np.int32)
                        # remapper的作用能将需要的语义标签映射到连续的整数值。
                        sem_labels = remapper[np.array(sem_labels)]

                        instance_labels = np.ascontiguousarray(block[:, -1])
                        # 得到实例标签转为浮点型数据
                        instance_labels = instance_labels.astype(np.float32)

                        disableInstanceBySemantic_labels = np.ascontiguousarray(block[:, -2])
                        disableInstanceBySemantic_labels = disableInstanceBySemantic_labels.astype(np.int32)
                        # 将禁用实例的语义标签映射到特定的新标签值。
                        disableInstanceBySemantic_labels = remapper_disableInstanceBySemantic[
                            np.array(disableInstanceBySemantic_labels)
                        ]
                        # 根据禁用实例的语义标签，将实例标签中对应的实例值更改为-100，即将禁用的实例标签设为特殊值。
                        instance_labels = np.where(disableInstanceBySemantic_labels == -100, -100, instance_labels)

                        # map instance from 0.
                        # [1:] because there are -100
                        # 计算实例标签中唯一的实例值，排除特殊值-100，并将其转换为整数类型。
                        uniqueInstances = (np.unique(instance_labels))[1:].astype(np.int32)
                        # 建一个长度为50000的列表，用于将实例标签映射到连续的整数值。
                        remapper_instance = np.ones(50000) * (-100)
                        # 遍历唯一实例值，并将其映射到连续的整数值。这样做可以确保实例标签按照顺序映射到从0开始的连续整数。
                        for i, j in enumerate(uniqueInstances):
                            remapper_instance[j] = i

                        instance_labels = remapper_instance[instance_labels.astype(np.int32)]

                        uniqueSemantics = (np.unique(sem_labels))[1:].astype(np.int32)

                        if split == "train" and (
                            len(uniqueInstances) < 10 or (len(uniqueSemantics) >= (len(uniqueInstances) - 2))
                        ):
                            print("unique insance: %d" % len(uniqueInstances))
                            print("unique semantic: %d" % len(uniqueSemantics))
                            print()
                            counter += 1
                        else:
                            torch.save((coords, colors, sem_labels, instance_labels), outFilePath)
                    else:
                        torch.save((coords, colors), outFilePath)
    print("Total skipped file :%d" % counter)
    json.dump(coordShift, open(outJsonPath, "w"))


if __name__ == "__main__":
    data_folder = "Synthetic_v3_InstanceSegmentation"
    filesOri = sorted(glob.glob(data_folder + "/*.txt"))

    trainSplit = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
    trainFiles = getFiles(filesOri, trainSplit)
    split = "train"
    trainOutDir = split
    os.makedirs(trainOutDir, exist_ok=True)
    preparePthFiles(trainFiles, split, trainOutDir, AugTimes=6)

    valSplit = [5, 10, 15, 20, 25]
    split = "val_250m"
    valFiles = getFiles(filesOri, valSplit)
    valOutDir = split
    os.makedirs(valOutDir, exist_ok=True)
    preparePthFiles(valFiles, split, valOutDir, crop_size=250)
