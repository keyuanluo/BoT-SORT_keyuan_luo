import os
import pandas as pd


def process_excel_file(input_excel_path):
    # 读取Excel文件到DataFrame
    df = pd.read_excel(input_excel_path)

    # 按照'image_name'分组
    grouped = df.groupby('image_name')

    # 创建一个空的列表来存储新的行
    new_rows = []

    # 遍历每一组（即每一帧）
    for image_name, group in grouped:
        # 初始化新行的数据
        new_row = {'image_name': image_name}

        # 获取当前帧的所有 track_id
        ids_in_frame = group['track_id'].unique()

        # 处理 track_id 为 1 的数据
        if 1 in ids_in_frame:
            row_id1 = group[group['track_id'] == 1].iloc[0]
            new_row['x1'] = row_id1['x1']
            new_row['y1'] = row_id1['y1']
        else:
            # 如果不存在，设置为 None 或者您指定的默认值
            new_row['x1'] = None
            new_row['y1'] = None

        # 处理 track_id 为 5 的数据
        if 5 in ids_in_frame:
            row_id5 = group[group['track_id'] == 5].iloc[0]
            new_row['x2'] = row_id5['x2']
            new_row['y2'] = row_id5['y2']
        else:
            # 如果不存在，设置为 None 或者您指定的默认值
            new_row['x2'] = None
            new_row['y2'] = None

        # 将新行添加到列表中
        new_rows.append(new_row)

    # 将新行列表转换为 DataFrame
    new_df = pd.DataFrame(new_rows)

    # 获取输入文件的文件夹路径和文件名
    input_folder = os.path.dirname(input_excel_path)
    input_filename = os.path.basename(input_excel_path)

    # 创建输出文件名，添加 '_processed' 后缀
    output_filename = os.path.splitext(input_filename)[0] + '_processed.xlsx'
    output_excel_path = os.path.join(input_folder, output_filename)

    # 保存新的 DataFrame 到 Excel 文件
    new_df.to_excel(output_excel_path, index=False)

    print(f'处理后的数据已保存到 {output_excel_path}')


# 示例使用
input_excel_path = '/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box/cloudy/01_06_2022_18_42_35_scene1_walk_cross_veryclose6/detections_1und5.xlsx'  # 输入的 Excel 文件路径

process_excel_file(input_excel_path)
