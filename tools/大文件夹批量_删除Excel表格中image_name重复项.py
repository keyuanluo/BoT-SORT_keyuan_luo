# import os
# import pandas as pd
#
# # 定义输入大文件夹路径
# base_folder = "/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box"
#
# # 遍历大文件夹下的11个子文件夹
# for subfolder in os.listdir(base_folder):
#     subfolder_path = os.path.join(base_folder, subfolder)
#
#     # 检查是否为目录
#     if os.path.isdir(subfolder_path):
#         # 遍历子文件夹下的所有子子文件夹
#         for sub_subfolder in os.listdir(subfolder_path):
#             sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)
#
#             # 检查是否为目录
#             if os.path.isdir(sub_subfolder_path):
#                 input_file = os.path.join(sub_subfolder_path, 'detections_最终.xlsx')
#                 output_file = os.path.join(sub_subfolder_path, 'deleted_same_image_name.xlsx')
#
#                 # 检查 Excel 文件是否存在
#                 if os.path.exists(input_file):
#                     print(f"正在处理: {input_file}")
#
#                     # 读取Excel文件
#                     df = pd.read_excel(input_file)
#
#                     # 统计原始行数
#                     original_count = len(df)
#
#                     # 删除image_name列中的重复项，只保留第一次出现的行
#                     df_unique = df.drop_duplicates(subset=['image_name'], keep='first')
#
#                     # 统计去重后的行数
#                     unique_count = len(df_unique)
#
#                     # 计算删除的数量
#                     deleted_count = original_count - unique_count
#
#                     # 保存去重后的数据到新的Excel文件
#                     df_unique.to_excel(output_file, index=False)
#
#                     print(f"处理完成，已删除 {deleted_count} 条重复项，去重后的文件已保存到：{output_file}")
#                 else:
#                     print(f"未找到 Excel 文件：{input_file}，跳过该文件夹。")
#
# print("所有子子文件夹处理完成！")

import os
import pandas as pd

# 定义输入大文件夹路径
base_folder = "/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box"

# 记录存在重复项的 Excel 文件信息
duplicate_files_info = []

# 遍历大文件夹下的11个子文件夹
for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)

    # 检查是否为目录
    if os.path.isdir(subfolder_path):
        # 遍历子文件夹下的所有子子文件夹
        for sub_subfolder in os.listdir(subfolder_path):
            sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)

            # 检查是否为目录
            if os.path.isdir(sub_subfolder_path):
                input_file = os.path.join(sub_subfolder_path, 'detections_最终.xlsx')
                output_file = os.path.join(sub_subfolder_path, 'deleted_same_image_name.xlsx')

                # 检查 Excel 文件是否存在
                if os.path.exists(input_file):
                    print(f"正在处理: {input_file}")

                    # 读取Excel文件
                    df = pd.read_excel(input_file)

                    # 统计原始行数
                    original_count = len(df)

                    # 删除image_name列中的重复项，只保留第一次出现的行
                    df_unique = df.drop_duplicates(subset=['image_name'], keep='first')

                    # 统计去重后的行数
                    unique_count = len(df_unique)

                    # 计算删除的数量
                    deleted_count = original_count - unique_count

                    # 保存去重后的数据到新的Excel文件
                    df_unique.to_excel(output_file, index=False)

                    if deleted_count > 0:
                        print(f"处理完成，已删除 {deleted_count} 条重复项，去重后的文件已保存到：{output_file}")
                        # 记录文件信息
                        duplicate_files_info.append({
                            "原始文件路径": input_file,
                            "去重后文件路径": output_file,
                            "删除的重复项数": deleted_count
                        })
                    else:
                        print(f"未找到重复项：{input_file}")
                else:
                    print(f"未找到 Excel 文件：{input_file}，跳过该文件夹。")

# 定义 Excel 记录文件的路径
summary_excel_path = "/media/robert/4TB-SSD/重复项记录.xlsx"

# 如果有需要记录的文件，生成 Excel 汇总表
if duplicate_files_info:
    df_summary = pd.DataFrame(duplicate_files_info)
    df_summary.to_excel(summary_excel_path, index=False)
    print(f"已生成 Excel 记录文件：{summary_excel_path}")
else:
    print("未找到任何存在重复项的文件，不生成 Excel 文档。")

print("所有子子文件夹处理完成！")

