import os
import random
import numpy as np
import torch
import pandas as pd
import math


'''
该脚本用来处理点云文件，将其生成对应的.pth文件，
将一个txt点云文件文件夹划分到两个文件夹，比例为8：2
分别是训练集和验证集
'''
def load_txt_data(txt_file):
    # 从txt文件中加载点云数据

    data = np.genfromtxt(txt_file, delimiter=',')
    coords = data[:, :3]  # 前三列是坐标
    colors = data[:, 3:6]  # 接下来三列是颜色
    sem_labels = data[:, -2]  # 语义标签
    instance_labels = data[:, -1]  # 实例标签

    return coords, colors, sem_labels, instance_labels

def save_as_pth(coords, colors, sem_labels, instance_labels, pth_file):
    # 将数据保存为.pth文件
    coords = np.ascontiguousarray(coords)
    colors = np.ascontiguousarray(colors)/ 127.5 - 1
    sem_labels = np.ascontiguousarray(sem_labels)
    instance_labels = np.ascontiguousarray(instance_labels)

    torch.save((coords, colors, sem_labels, instance_labels), pth_file)
    print(f"数据已保存为.pth文件：{pth_file}")


# semanticKeep用来表示需要增强的类
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

if __name__ == "__main__":
    data_folder = "tea_for_train"  # 替换为存放txt文件的文件夹路径
    train_ratio = 0.8  # 训练集占比
    random_seed = 42  # 随机种子，保证每次运行结果相同

    # 获取所有txt文件的路径列表
    txt_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".txt")]

    # 设置随机种子，保证每次分割结果相同
    random.seed(random_seed)
    random.shuffle(txt_files)

    # 计算训练集和测试集的切分点
    train_split_idx = int(len(txt_files) * train_ratio)

    # 分割训练集和测试集
    train_files = txt_files[:train_split_idx]
    test_files = txt_files[train_split_idx:]

    # 保存训练集数据为.pth文件
    train_out_folder = "tea_for_train_pth"  # 替换为保存训练集.pth文件的文件夹路径
    os.makedirs(train_out_folder, exist_ok=True)

    for txt_file in train_files:
        coords, colors, sem_labels, instance_labels = load_txt_data(txt_file)
        pth_file = os.path.join(train_out_folder, os.path.splitext(os.path.basename(txt_file))[0] + ".pth")
        save_as_pth(coords, colors, sem_labels, instance_labels, pth_file)

    print("训练集数据已保存为.pth文件！")

    # 保存测试集数据为.pth文件
    test_out_folder = "tea_for_test_pth"  # 替换为保存测试集.pth文件的文件夹路径
    os.makedirs(test_out_folder, exist_ok=True)

    for txt_file in test_files:
        coords, colors, _, _ = load_txt_data(txt_file)
        pth_file = os.path.join(test_out_folder, os.path.splitext(os.path.basename(txt_file))[0] + ".pth")
        save_as_pth(coords, colors, None, None, pth_file)

    print("测试集数据已保存为.pth文件！")


