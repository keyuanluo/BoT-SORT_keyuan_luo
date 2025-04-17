import os
import pandas as pd


def process_excel_file(input_excel_path):
    # 读取Excel文件到DataFrame
    df = pd.read_excel(input_excel_path)

    # 创建一个空的列表来存储新的行
    new_rows = []

    # 按照'image_name'分组
    grouped = df.groupby('image_name')

    # 遍历每一组（即每一帧）
    for image_name, group in grouped:
        ids_in_frame = group['track_id'].unique()
        if 1 in ids_in_frame and 5 in ids_in_frame:
            # 获取track_id为1和5的行
            row_id1 = group[group['track_id'] == 1].iloc[0]
            row_id5 = group[group['track_id'] == 5].iloc[0]

            # 创建一个新的行，包含需要的值
            new_row = {
                'image_name': image_name,
                'x1': row_id1['x1'],
                'y1': row_id1['y1'],
                'x2': row_id5['x2'],
                'y2': row_id5['y2']
            }
            # 将新行添加到列表中
            new_rows.append(new_row)
        else:
            # 如果id1或id5不存在，可以选择是否处理或跳过
            # 这里我们选择跳过
            pass

    # 将新行列表转换为DataFrame
    new_df = pd.DataFrame(new_rows)

    # 获取输入文件的文件夹路径和文件名
    input_folder = os.path.dirname(input_excel_path)
    input_filename = os.path.basename(input_excel_path)

    # 创建输出文件名，添加 '_processed' 后缀
    output_filename = os.path.splitext(input_filename)[0] + '_processed.xlsx'
    output_excel_path = os.path.join(input_folder, output_filename)

    # 保存新的DataFrame到Excel文件
    new_df.to_excel(output_excel_path, index=False)

    print(f'处理后的数据已保存到 {output_excel_path}')


# 示例使用
input_excel_path = '/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box/cloudy/01_06_2022_18_42_35_scene1_walk_cross_veryclose6/detections_1und5.xlsx'  # 输入的Excel文件路径

process_excel_file(input_excel_path)
