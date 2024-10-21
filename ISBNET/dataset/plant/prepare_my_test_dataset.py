import os
import random
import numpy as np
import torch


'''
该脚本用来将测试文件夹内的所有的txt文件
都生成成对应的.pth文件
'''
def load_txt_data(txt_file):
    # 从txt文件中加载点云数据

    data = np.genfromtxt(txt_file,delimiter=',')
    coords = data[:, :3]  # 前三列是坐标
    colors = data[:, 3:6]  # 接下来三列是颜色


    return coords, colors

def save_as_pth(coords, colors,  pth_file):
    # 将数据保存为.pth文件
    coords = np.ascontiguousarray(coords)
    colors = np.ascontiguousarray(colors)/ 127.5 - 1


    torch.save((coords, colors), pth_file)
    print(f"数据已保存为.pth文件：{pth_file}")

if __name__ == "__main__":
    data_folder = "tea-train"  # 替换为存放txt文件的文件夹路径
    # 获取所有txt文件的路径列表
    txt_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".txt")]

    # 保存测试集数据为.pth文件
    test_out_folder = "tea_train_for_train_pth"  # 替换为保存测试集.pth文件的文件夹路径
    os.makedirs(test_out_folder, exist_ok=True)

    for txt_file in txt_files:
        coords, colors = load_txt_data(txt_file)
        pth_file = os.path.join(test_out_folder, os.path.splitext(os.path.basename(txt_file))[0] + ".pth")
        save_as_pth(coords, colors,  pth_file)

    print("测试集数据已保存为.pth文件！")
