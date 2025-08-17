import os
import laspy


def convert_xyz_to_laz(input_dir, output_subdir_name="sampled_output"):
    """
    将指定目录下的所有`.xyz`文件转换为`.laz`文件，并保存到一个新子目录中。

    :param input_dir: 输入文件夹路径，包含待处理的`.xyz`文件。
    :param output_subdir_name: 输出文件夹的子目录名称，默认名为`processed_laz_files`。
    """
    # 创建新的子目录用于保存输出文件
    output_dir = os.path.join(input_dir, output_subdir_name)
    os.makedirs(output_dir, exist_ok=True)  # 如果子目录不存在则创建

    # 遍历目录下的所有文件
    for file_name in os.listdir(input_dir):
        # 检查文件是否以`.xyz`为后缀
        if file_name.endswith('.xyz'):
            input_file_path = os.path.join(input_dir, file_name)
            output_file_name = os.path.splitext(file_name)[0] + ".laz"  # 替换后缀为.laz
            output_file_path = os.path.join(output_dir, output_file_name)

            try:
                # 读取原始`.xyz`文件内容
                points = []
                with open(input_file_path, 'r') as xyz_file:
                    for line in xyz_file:
                        if line.strip():  # 忽略空行
                            x, y, z = map(float, line.strip().split()[:3])  # 只取前三列
                            points.append((x, y, z))

                # 创建 LAS 文件对象
                header = laspy.LasHeader(point_format=3, version="1.2")
                las = laspy.LasData(header)

                # 添加点云数据
                las.x, las.y, las.z = zip(*points)

                # 写入到 .laz 文件
                las.write(output_file_path)
                print(f"成功转换文件: {input_file_path} -> {output_file_path}")
            except Exception as e:
                print(f"处理文件 {input_file_path} 时出错: {e}")

    print(f"所有文件已处理完毕，输出目录: {output_dir}")


# 主函数
if __name__ == "__main__":
    # 设置输入目录路径
    input_directory = r"E:\mybishe\dataset\TreeNet\TreeNet3D\myuse\smallLeaf"  # 替换为实际路径

    # 调用处理函数
    convert_xyz_to_laz(input_directory)
