# import os
# import numpy as np
#
# def modify_fourth_column(input_directory):
#     # 在输入目录中创建一个新文件夹，命名为 modified_txt_files
#     output_directory = os.path.join(input_directory, "modified_txt_files")
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#
#     # 遍历输入目录中的所有 .txt 文件
#     for filename in os.listdir(input_directory):
#         if filename.endswith('.txt'):
#             input_file_path = os.path.join(input_directory, filename)
#             output_file_path = os.path.join(output_directory, filename)
#
#             # 读取 txt 文件中的数据
#             data = np.loadtxt(input_file_path)
#
#             # 修改第四列的值，将 0 替换为 2
#             data[:, 3] = np.where(data[:, 3] == 0, 6, data[:, 3])
#
#             # 将修改后的数据保存到输出文件夹中的新文件
#             np.savetxt(output_file_path, data, fmt="%.6f %.6f %.6f %d")
#             print(f"文件已保存到: {output_file_path}")
#
# # 使用示例
# input_directory = "E:\\dataset\\tree\\singletree_ULS\\class_inhance\\QueRub\\sampled_output\\sampled_output"  # 输入目录路径
# modify_fourth_column(input_directory)


import os
import numpy as np

def modify_fourth_column(input_directory):
    # 在输入目录中创建一个新文件夹，命名为 modified_txt_files
    output_directory = os.path.join(input_directory, "txt_file_add10")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 遍历输入目录中的所有 .txt 文件
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename)

            # 读取 txt 文件中的数据
            data = np.loadtxt(input_file_path)

            # # 修改第四列的值，根据第四列的值进行替换
            # # 如果第四列的值为 0，修改为 10；如果为 1，修改为 11；依此类推
            # data[:, 3] = np.where(data[:, 3] >= 0, data[:, 3] + 10, data[:, 3])

            mapping = {
                0: 23,
                1: 22,
                2: 19,
                3: 20,
                # 继续添加需要映射的值
            }
            # 使用 NumPy 向量化替换：如果值在 mapping 字典中，则替换，否则保持原值
            data[:, 3] = np.vectorize(lambda x: mapping.get(x, x))(data[:, 3])


            # 将修改后的数据保存到输出文件夹中的新文件
            np.savetxt(output_file_path, data, fmt="%.6f %.6f %.6f %d")
            print(f"文件已保存到: {output_file_path}")

# 使用示例
input_directory = r"E:\mybishe\dataset\SYSSIFOSS\singletree_ULS\addition\PruAvi\sampled_output"  # 输入目录路径
modify_fourth_column(input_directory)
