import os
import laspy
import numpy as np

# 配置参数
input_file = r"E:\mybishe\dataset\Ours\CAR\yinxingdadao\yinxingdadao_part2_seg_second_with_label.txt"  # 输入点云文件
output_folder = r"E:\mybishe\dataset\Ours\CAR\processed"  # 输出文件夹
n = 6  # 指定标签所在的列（从 0 开始计数，例如第 7 列，n=6）
treename = "yinxing(2)"  # 文件前缀，可调

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 存储不同标签的点
point_dict = {}

# 记录有问题的类标签（仅记录 NaN 或 非数值数据）
problematic_labels = set()

# 读取点云文件
with open(input_file, 'r') as f:
    for line in f:
        data = line.strip().split()  # 按空格分割数据

        # 确保数据长度足够
        if len(data) <= n:
            continue  # 跳过该行

        # 尝试转换为数值
        try:
            float_data = [float(x) for x in data]
        except ValueError:
            # 如果出现非数值数据，记录该类标签
            problematic_labels.add(data[n])
            continue  # 跳过该行

        # 获取分类标签
        label = int(float_data[n])

        # 存储数据（无论 classification 是否超范围，都正常处理）
        label_str = str(label)
        if label_str not in point_dict:
            point_dict[label_str] = []
        point_dict[label_str].append(float_data)

# 输出存在问题的类标签（仅包含 NaN 或 非数值数据的标签）
if problematic_labels:
    print(f"⚠️ 存在非数值数据的类标签: {sorted(problematic_labels)}")


# 解析点云数据格式
def parse_point_data(points):
    """ 解析点云数据，并处理 NaN/无效值 """
    points_array = np.array(points, dtype=float)  # 转换为浮点数
    points_array = np.nan_to_num(points_array, nan=0.0)  # 处理 NaN，将 NaN 转换为 0

    classification_values = points_array[:, n].astype(int)

    # 限制 classification 值范围（0-31）
    classification_values = np.clip(classification_values, 0, 31)

    return {
        "x": points_array[:, 0],
        "y": points_array[:, 1],
        "z": points_array[:, 2],
        "intensity": points_array[:, 3] if points_array.shape[1] > 3 else np.zeros(len(points)),  # 处理无强度数据情况
        "classification": classification_values
    }


# 按标签保存不同的点云数据
for label, points in point_dict.items():
    output_file = os.path.join(output_folder, f"{treename}{label}.laz")

    # 解析点云数据
    parsed_data = parse_point_data(points)

    # 创建 LAS/LAZ 文件（使用 LAS 1.2 避免兼容性问题）
    header = laspy.LasHeader(point_format=1, version="1.2")
    las = laspy.LasData(header)

    # 赋值
    las.x = parsed_data["x"]
    las.y = parsed_data["y"]
    las.z = parsed_data["z"]
    las.intensity = parsed_data["intensity"]
    las.classification = parsed_data["classification"]

    # 保存为 LAZ 格式
    las.write(output_file)

print(f"✅ 数据处理完成，LAZ 文件已保存在 {output_folder} 文件夹下！")