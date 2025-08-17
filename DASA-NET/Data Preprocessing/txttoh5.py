# import os
# import numpy as np
# import h5py
#
#
# def load_txt_file(file_path):
#     """
#     加载一个点云 txt 文件，返回 xyz 坐标和标签
#
#     参数:
#     file_path (str): txt 文件的路径
#
#     返回:
#     tuple: (xyz 坐标数组, 标签值)
#     """
#     data = np.loadtxt(file_path)  # 加载数据
#     xyz = data[:, :3]  # 提取前三列为 xyz 坐标
#     label = int(data[0, 3])  # 假设每个文件的所有标签值都是相同的，取第一行的标签
#     return xyz, label
#
#
# def process_directory_to_hdf5(directory):
#     """
#     将指定目录下的所有点云 txt 文件转换为一个 HDF5 文件
#
#     参数:
#     directory (str): 包含点云 txt 文件的目录
#     """
#     output_dir = os.path.join(directory, "h5_file")
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     output_hdf5_file = os.path.join(output_dir, "pointclouds.h5")
#     txt_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
#
#     all_data = []
#     all_labels = []
#
#     for file_path in txt_files:
#         print(f"正在处理文件: {file_path}")
#         xyz, label = load_txt_file(file_path)
#
#         if xyz.shape[0] != 2048:
#             print(f"警告: 文件 {file_path} 的点数量不是 2048，已跳过。")
#             continue
#
#         all_data.append(xyz)
#         all_labels.append([label])  # 将标签包装成列表，确保形状为 (N, 1)
#
#     all_data = np.array(all_data, dtype=np.float64)  # 形状 (N, 2048, 3) 类型为 float64
#     all_labels = np.array(all_labels, dtype=np.int64)  # 形状 (N, 1)
#
#     with h5py.File(output_hdf5_file, 'w') as f:
#         f.create_dataset('data', data=all_data)
#         f.create_dataset('label', data=all_labels)
#         print(f"HDF5 文件已保存到: {output_hdf5_file}")
#
#     split_data_by_label(output_hdf5_file, output_dir)
#
#
# def split_data_by_label(h5_file_path, output_dir):
#     """
#     基于 pointclouds.h5 文件的数据，将每个标签类中的 70% 数据写入 ply_data_train0.h5，
#     剩余 30% 写入 ply_data_test0.h5。
#
#     参数:
#     h5_file_path (str): 输入 H5 文件的路径
#     output_dir (str): 输出 H5 文件的目录
#     """
#     with h5py.File(h5_file_path, 'r') as f:
#         all_data = f['data'][:]
#         all_labels = f['label'][:]
#
#         # 获取唯一标签
#         unique_labels = np.unique(all_labels)
#         train_data = []
#         train_labels = []
#         test_data = []
#         test_labels = []
#
#         # 按标签分组
#         for label in unique_labels:
#             indices = np.where(all_labels == label)[0]
#             np.random.shuffle(indices)  # 随机打乱索引
#
#             split_point = int(len(indices) * 0.7)
#             train_indices = indices[:split_point]
#             test_indices = indices[split_point:]
#
#             train_data.append(all_data[train_indices])
#             train_labels.append(all_labels[train_indices])
#             test_data.append(all_data[test_indices])
#             test_labels.append(all_labels[test_indices])
#
#         # 合并所有训练和测试数据
#         train_data = np.concatenate(train_data, axis=0)
#         train_labels = np.concatenate(train_labels, axis=0)
#         test_data = np.concatenate(test_data, axis=0)
#         test_labels = np.concatenate(test_labels, axis=0)
#
#         # 创建 ply_data_train0.h5 和 ply_data_test0.h5 文件
#         train_file_path = os.path.join(output_dir, "ply_data_train0.h5")
#         test_file_path = os.path.join(output_dir, "ply_data_test0.h5")
#
#         with h5py.File(train_file_path, 'w') as train_f:
#             train_f.create_dataset('data', data=train_data)
#             train_f.create_dataset('label', data=train_labels)
#             print(f"训练数据 HDF5 文件已保存到: {train_file_path}")
#
#         with h5py.File(test_file_path, 'w') as test_f:
#             test_f.create_dataset('data', data=test_data)
#             test_f.create_dataset('label', data=test_labels)
#             print(f"测试数据 HDF5 文件已保存到: {test_file_path}")
#
#
# # 用法示例
# directory_path = "E:\\dataset\\tree\\singletree_ULS\\on_txt_to_h5"  # 包含 txt 文件的目录
# process_directory_to_hdf5(directory_path)


import os
import numpy as np
import h5py


def load_txt_file(file_path):
    """
    加载一个点云 txt 文件，返回 xyz 坐标和标签

    参数:
    file_path (str): txt 文件的路径

    返回:
    tuple: (xyz 坐标数组, 标签值)
    """
    data = np.loadtxt(file_path)  # 加载数据
    xyz = data[:, :3]  # 提取前三列为 xyz 坐标
    label = int(data[0, 3])  # 假设每个文件的所有标签值都是相同的，取第一行的标签
    return xyz, label


def process_directory_to_hdf5(directory):
    """
    将指定目录下的所有点云 txt 文件转换为一个 HDF5 文件

    参数:
    directory (str): 包含点云 txt 文件的目录
    """
    output_dir = os.path.join(directory, "h5_file")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_hdf5_file = os.path.join(output_dir, "pointclouds.h5")
    txt_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]

    all_data = []
    all_labels = []

    for file_path in txt_files:
        print(f"正在处理文件: {file_path}")
        xyz, label = load_txt_file(file_path)

        if xyz.shape[0] != 2048:
            print(f"警告: 文件 {file_path} 的点数量不是 2048，已跳过。")
            continue

        all_data.append(xyz)
        all_labels.append([label])  # 将标签包装成列表，确保形状为 (N, 1)

    all_data = np.array(all_data, dtype=np.float64)  # 形状 (N, 2048, 3) 类型为 float64
    all_labels = np.array(all_labels, dtype=np.int64)  # 形状 (N, 1)

    with h5py.File(output_hdf5_file, 'w') as f:
        f.create_dataset('data', data=all_data)
        f.create_dataset('label', data=all_labels)
        print(f"HDF5 文件已保存到: {output_hdf5_file}")

    split_data_by_label(output_hdf5_file, output_dir)


def split_data_by_label(h5_file_path, output_dir, random_seed=42, max_files_per_h5=2048):

    np.random.seed(random_seed)  # 在此处设置随机种子
    with h5py.File(h5_file_path, 'r') as f:
        all_data = f['data'][:]
        all_labels = f['label'][:]

        unique_labels = np.unique(all_labels)
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        train_file_count = 0
        test_file_count = 0

        # 按标签分组
        for label in unique_labels:
            indices = np.where(all_labels == label)[0]
            np.random.shuffle(indices)  # 随机打乱索引

            split_point = int(len(indices) * 0.7)
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]

            train_data.append(all_data[train_indices])
            train_labels.append(all_labels[train_indices])
            test_data.append(all_data[test_indices])
            test_labels.append(all_labels[test_indices])

        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        test_data = np.concatenate(test_data, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        # 按 2048 文件数拆分并存储 ply_data_train 和 ply_data_test 文件
        for i in range(0, len(train_data), max_files_per_h5):
            train_file_path = os.path.join(output_dir, f"ply_data_train{i // max_files_per_h5}.h5")
            with h5py.File(train_file_path, 'w') as train_f:
                train_f.create_dataset('data', data=train_data[i:i + max_files_per_h5])
                train_f.create_dataset('label', data=train_labels[i:i + max_files_per_h5])
            print(f"训练数据 HDF5 文件已保存到: {train_file_path}")

        for j in range(0, len(test_data), max_files_per_h5):
            test_file_path = os.path.join(output_dir, f"ply_data_test{j // max_files_per_h5}.h5")
            with h5py.File(test_file_path, 'w') as test_f:
                test_f.create_dataset('data', data=test_data[j:j + max_files_per_h5])
                test_f.create_dataset('label', data=test_labels[j:j + max_files_per_h5])
            print(f"测试数据 HDF5 文件已保存到: {test_file_path}")


# 用法示例
directory_path = r"E:\mybishe\dataset\final_data2"  # 包含 txt 文件的目录
process_directory_to_hdf5(directory_path)
