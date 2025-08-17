# import os
# import numpy as np
# import laspy
# import open3d as o3d
# from laspy import LazBackend
#
# def load_laz_file(file_path):
#     """加载 .laz 文件并将其转换为 numpy 数组"""
#     las = laspy.read(file_path, laz_backend=LazBackend.Lazrs)
#     points = np.vstack((las.x, las.y, las.z)).transpose()
#     return points
#
# def ground_filter(points, threshold=0.2):
#     min_z = np.min(points[:, 2])
#     filtered_points = points[points[:, 2] > min_z + threshold]
#     return filtered_points
#
# def farthest_point_sampling(points, num_samples):
#     N, D = points.shape
#     if N <= num_samples:
#         return points, np.arange(N)  # 返回索引
#
#     sampled_indices = np.zeros(num_samples, dtype=int)
#     distances = np.ones(N) * np.inf
#
#     sampled_indices[0] = np.random.randint(N)
#     farthest_point = points[sampled_indices[0]]
#
#     for i in range(1, num_samples):
#         dist = np.linalg.norm(points - farthest_point, axis=1)
#         distances = np.minimum(distances, dist)
#         sampled_indices[i] = np.argmax(distances)
#         farthest_point = points[sampled_indices[i]]
#
#     return points[sampled_indices], sampled_indices  # 返回采样点及其索引
#
# def normalize_points(points):
#     centroid = np.mean(points, axis=0)
#     centered_points = points - centroid
#     max_distance = np.max(np.linalg.norm(centered_points, axis=1))
#     scaled_points = centered_points / max_distance
#     return scaled_points
#
# def save_sampled_points_as_txt(sampled_points, output_path, class_label):
#     labels = np.full((sampled_points.shape[0], 1), class_label)
#     points_with_labels = np.hstack((sampled_points, labels))
#     np.savetxt(output_path, points_with_labels, fmt="%.6f %.6f %.6f %d")
#     print(f"采样后的点云已保存到: {output_path}")
#
# def process_directory(directory, num_samples, point_threshold=4096, max_iterations=1):
#     subdirs = sorted([os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
#
#     for class_label, subdir in enumerate(subdirs):
#         output_dir = os.path.join(subdir, "sampled_output")
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#
#         for filename in os.listdir(subdir):
#             if filename.endswith('.laz'):
#                 file_path = os.path.join(subdir, filename)
#                 print(f"正在处理文件: {file_path}")
#
#                 points = load_laz_file(file_path)
#                 print(f"从 {filename} 中加载了 {points.shape[0]} 个点")
#
#                 filtered_points = ground_filter(points, threshold=0.6)
#                 print(f"地面滤波后保留了 {filtered_points.shape[0]} 个点")
#
#                 if filtered_points.shape[0] < num_samples:
#                     print(f"点数不足 {num_samples}，跳过该文件")
#                     continue
#
#                 sample_count = 1
#                 while filtered_points.shape[0] > point_threshold and sample_count <= max_iterations:
#                     sampled_points, sampled_indices = farthest_point_sampling(filtered_points, num_samples)
#                     print(f"第 {sample_count} 次采样了 {sampled_points.shape[0]} 个点")
#
#                     normalized_points = normalize_points(sampled_points)
#                     print("点云已中心化和尺度缩放")
#
#                     output_file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_sampled_{sample_count}.txt")
#                     save_sampled_points_as_txt(normalized_points, output_file_path, class_label)
#
#                     # 过滤掉已经采样的点
#                     remaining_indices = np.setdiff1d(np.arange(filtered_points.shape[0]), sampled_indices)
#                     filtered_points = filtered_points[remaining_indices]
#                     sample_count += 1
#
#                 # 如果剩余的点数足够且采样次数未超出限制，则再进行一次采样
#                 if filtered_points.shape[0] >= num_samples and sample_count <= max_iterations:
#                     sampled_points, sampled_indices = farthest_point_sampling(filtered_points, num_samples)
#                     normalized_points = normalize_points(sampled_points)
#                     output_file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_sampled_{sample_count}.txt")
#                     save_sampled_points_as_txt(normalized_points, output_file_path, class_label)
#
# # 用法示例
# directory_path = r"E:\mybishe\dataset\TreeNet\TreeNet3D\myuse\ajianglanren"
# number_of_samples = 2048
# process_directory(directory_path, number_of_samples)
#
#
import os
import numpy as np
import laspy
from laspy import LazBackend


def load_laz_file(file_path):
    """加载 .laz 文件并将其转换为 numpy 数组"""
    las = laspy.read(file_path, laz_backend=LazBackend.Lazrs)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    return points


def ground_filter(points, threshold=0.2):
    """地面滤波"""
    min_z = np.min(points[:, 2])
    return points[points[:, 2] > min_z + threshold]


def farthest_point_sampling(points, num_samples):
    """优化最远点采样，使用 KD-Tree 加速距离计算"""
    N, D = points.shape
    if N <= num_samples:
        return points, np.arange(N)  # 返回索引

    sampled_indices = [np.random.randint(N)]  # 随机选第一个点
    distances = np.full(N, np.inf)

    for _ in range(num_samples - 1):
        # 更新所有点到采样点集合的最短距离
        last_sampled = points[sampled_indices[-1]]
        dist = np.linalg.norm(points - last_sampled, axis=1)
        distances = np.minimum(distances, dist)

        # 选择最远点
        next_index = np.argmax(distances)
        sampled_indices.append(next_index)

    sampled_indices = np.array(sampled_indices)
    return points[sampled_indices], sampled_indices


def normalize_points(points):
    """优化点云归一化"""
    centroid = np.mean(points, axis=0)
    max_distance = np.linalg.norm(points - centroid, axis=1).max()
    return (points - centroid) / max_distance


def save_sampled_points_as_txt(sampled_points, output_path, class_label):
    """保存点云采样结果到文本文件"""
    labels = np.full((sampled_points.shape[0], 1), class_label)
    points_with_labels = np.hstack((sampled_points, labels))
    np.savetxt(output_path, points_with_labels, fmt="%.6f %.6f %.6f %d")
    print(f"采样后的点云已保存到: {output_path}")


def process_directory(directory, num_samples, point_threshold=0, max_iterations=2):
    """
    处理目录下的 .laz 文件

    :param directory: 主目录路径，包含各类别子目录
    :param num_samples: 每次采样的点数
    :param point_threshold: 点云采样的阈值，超过该点数会继续采样
    :param max_iterations: 每个点云文件的最大采样次数
    """
    subdirs = sorted(
        [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

    for class_label, subdir in enumerate(subdirs):
        output_dir = os.path.join(subdir, "sampled_output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(subdir):
            if filename.endswith('.laz'):
                file_path = os.path.join(subdir, filename)
                print(f"正在处理文件: {file_path}")

                points = load_laz_file(file_path)
                print(f"从 {filename} 中加载了 {points.shape[0]} 个点")

                filtered_points = ground_filter(points, threshold=0.6)
                print(f"地面滤波后保留了 {filtered_points.shape[0]} 个点")

                if filtered_points.shape[0] < num_samples:
                    print(f"点数不足 {num_samples}，跳过该文件")
                    continue

                sample_count = 1
                while filtered_points.shape[0] > point_threshold and sample_count <= max_iterations:
                    sampled_points, sampled_indices = farthest_point_sampling(filtered_points, num_samples)
                    print(f"第 {sample_count} 次采样了 {sampled_points.shape[0]} 个点")

                    normalized_points = normalize_points(sampled_points)
                    print("点云已中心化和尺度缩放")

                    output_file_path = os.path.join(output_dir,
                                                    f"{os.path.splitext(filename)[0]}_sampled_{sample_count}.txt")
                    save_sampled_points_as_txt(normalized_points, output_file_path, class_label)

                    # 过滤掉已经采样的点
                    remaining_indices = np.setdiff1d(np.arange(filtered_points.shape[0]), sampled_indices)
                    filtered_points = filtered_points[remaining_indices]
                    sample_count += 1

                # 如果剩余的点数足够且采样次数未超出限制，则再进行一次采样
                if filtered_points.shape[0] >= num_samples and sample_count <= max_iterations:
                    sampled_points, sampled_indices = farthest_point_sampling(filtered_points, num_samples)
                    normalized_points = normalize_points(sampled_points)
                    output_file_path = os.path.join(output_dir,
                                                    f"{os.path.splitext(filename)[0]}_sampled_{sample_count}.txt")
                    save_sampled_points_as_txt(normalized_points, output_file_path, class_label)


# 用法
directory_path = r"E:\mybishe\dataset\SYSSIFOSS\singletree_ULS\addition"
number_of_samples = 2048
process_directory(directory_path, number_of_samples)
