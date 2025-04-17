

import pandas as pd

# 输入和输出文件路径
input_file = "/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box/evening/12_08_2022_18_00_44_scene5_walk_stop_far/detections_最终.xlsx"  # 请替换为实际的Excel文件路径
output_file = "/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box/evening/12_08_2022_18_00_44_scene5_walk_stop_far/deleted_same_image_name.xlsx"

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

print(f"处理完成，已删除 {deleted_count} 条重复项，去重后的文件已保存到：{output_file}")
