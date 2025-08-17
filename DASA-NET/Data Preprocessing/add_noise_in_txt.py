# import os
# import numpy as np
# import random
#
# # 设置噪声参数
# noise_amplitude = 0.15  # 噪声的幅度，调整此值来改变抖动的程度
# noise_percentage = 1  # 添加噪声的点的百分比
#
#
# def add_noise_to_point_cloud(file_path, output_folder):
#     # 读取点云数据
#     data = np.loadtxt(file_path)
#
#     # 获取点云的数量
#     num_points = data.shape[0]
#
#     # 随机选取40%的点来添加噪声
#     num_noisy_points = int(num_points * noise_percentage)
#     noisy_indices = random.sample(range(num_points), num_noisy_points)
#
#     # 对选中的点添加随机噪声
#     noise = np.random.uniform(-noise_amplitude, noise_amplitude, size=(num_noisy_points, 3))  # 对x, y, z坐标添加噪声
#     data[noisy_indices, 0:3] += noise
#
#     # 确保输出文件夹存在
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 获取输出文件路径
#     output_path = os.path.join(output_folder, os.path.basename(file_path))
#
#     # 保存修改后的点云数据到新的文件夹
#     np.savetxt(output_path, data, fmt='%.6f', delimiter=' ')
#     print(f'噪声添加完毕，保存为: {output_path}')
#
#
# def process_point_cloud_files(input_folder, output_folder):
#     # 遍历输入文件夹中的所有txt文件
#     for filename in os.listdir(input_folder):
#         if filename.endswith('.txt'):
#             file_path = os.path.join(input_folder, filename)
#             add_noise_to_point_cloud(file_path, output_folder)
#
#
# # 设置文件夹路径
# input_folder = r'E:\mybishe\dataset\TreeNet\TreeNet3D\myuse\txttoh5_middle'  # 输入文件夹路径
# output_folder = r'E:\mybishe\dataset\TreeNet\TreeNet3D\myuse\txttoh5_middle_noise'  # 输出文件夹路径
#
# # 处理所有点云文件
# process_point_cloud_files(input_folder, output_folder)



import os
import numpy as np
import random

# 设置噪声参数
noise_std = 0.15  # 高斯噪声的标准差，调整此值来改变噪声的大小（幅度）
noise_percentage = 0.7  # 添加噪声的点的百分比

def add_noise_to_point_cloud(file_path, output_folder):
    # 读取点云数据
    data = np.loadtxt(file_path)

    # 获取点云的数量
    num_points = data.shape[0]

    # 随机选取需要添加噪声的点的数量
    num_noisy_points = int(num_points * noise_percentage)  # 噪声的点数
    noisy_indices = random.sample(range(num_points), num_noisy_points)

    # 对选中的点添加高斯噪声
    noise = np.random.normal(loc=0, scale=noise_std, size=(num_noisy_points, 3))  # 生成高斯噪声
    data[noisy_indices, 0:3] += noise  # 向x, y, z坐标添加噪声

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输出文件路径
    output_path = os.path.join(output_folder, os.path.basename(file_path))

    # 保存修改后的点云数据到新的文件夹
    np.savetxt(output_path, data, fmt='%.6f', delimiter=' ')
    print(f'噪声添加完毕，保存为: {output_path}')

def process_point_cloud_files(input_folder, output_folder):
    # 遍历输入文件夹中的所有txt文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            add_noise_to_point_cloud(file_path, output_folder)

# 设置文件夹路径
input_folder = r'E:\mybishe\dataset\TreeNet\TreeNet3D\myuse\txttoh5_middle'  # 输入文件夹路径
output_folder = r'E:\mybishe\dataset\TreeNet\TreeNet3D\myuse\txttoh5_middle_noise'  # 输出文件夹路径

# 处理所有点云文件
process_point_cloud_files(input_folder, output_folder)
