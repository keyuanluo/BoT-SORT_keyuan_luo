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
        # 初始化一个字典来存储需要处理的id对
        #id_pairs = [(1, 4), (5, 6)]
        id_pairs = [(1, 5)]

        # 处理每一对id
        for id_main, id_aux in id_pairs:
            ids_in_frame = group['track_id'].unique()
            if id_main in ids_in_frame and id_aux in ids_in_frame:
                # 获取id_main和id_aux的行
                row_main = group[group['track_id'] == id_main].iloc[0]
                row_aux = group[group['track_id'] == id_aux].iloc[0]

                # 创建一个新的行，复制id_main的行
                new_row = row_main.copy()
                # 将y1替换为id_aux的y1
                new_row['y1'] = row_aux['y1']

                # 将新行添加到列表中
                new_rows.append(new_row)
            else:
                # 如果不同时存在，保留原始的id_main的行
                if id_main in ids_in_frame:
                    row_main = group[group['track_id'] == id_main].iloc[0]
                    new_rows.append(row_main)

        # 处理未在id_pairs中的其他track_id（如果需要保留）
        # 如果您希望保留其他track_id的数据，请取消以下注释
        # for track_id in ids_in_frame:
        #     if track_id not in [id_main for id_main, _ in id_pairs]:
        #         row = group[group['track_id'] == track_id].iloc[0]
        #         new_rows.append(row)

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
