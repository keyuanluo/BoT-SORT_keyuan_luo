import pandas as pd
import numpy as np


def fill_missing_frames(df, track_id_list=None):
    # 提取帧号，假设 image_name 的格式为 '000370.png'
    df['frame_number'] = df['image_name'].str.extract('(\d+)').astype(int)

    # 如果未指定 track_id_list，则对所有的 track_id 进行处理
    if track_id_list is None:
        track_id_list = df['track_id'].unique()

    # 用于存储填充后的所有数据
    filled_df = pd.DataFrame()

    for track_id in track_id_list:
        # 筛选出当前 track_id 的数据，并按照帧号排序
        track_df = df[df['track_id'] == track_id].sort_values('frame_number').reset_index(drop=True)

        # 创建一个新的 DataFrame 来存储当前 track_id 的填充数据
        track_filled_df = pd.DataFrame()

        for idx in range(len(track_df) - 1):
            current_row = track_df.loc[idx]
            next_row = track_df.loc[idx + 1]

            # 计算当前帧和下一帧之间的差值
            frame_diff = next_row['frame_number'] - current_row['frame_number']

            # 将当前行添加到填充数据中
            track_filled_df = track_filled_df.append(current_row, ignore_index=True)

            # 如果帧差值大于1，表示存在缺失帧，需要进行插值
            if frame_diff > 1:
                for gap in range(1, frame_diff):
                    # 计算插值比例
                    ratio = gap / frame_diff

                    # 生成新的帧号
                    new_frame_number = current_row['frame_number'] + gap
                    new_image_name = '{:06d}.png'.format(new_frame_number)

                    # 对 x1, y1, x2, y2, score 进行线性插值
                    interpolated_row = {
                        'image_name': new_image_name,
                        'frame_gap': 1,  # 填充的帧 gap 设为 1
                        'track_id': track_id,
                        'class': current_row['class'],
                        'x1': current_row['x1'] + ratio * (next_row['x1'] - current_row['x1']),
                        'y1': current_row['y1'] + ratio * (next_row['y1'] - current_row['y1']),
                        'x2': current_row['x2'] + ratio * (next_row['x2'] - current_row['x2']),
                        'y2': current_row['y2'] + ratio * (next_row['y2'] - current_row['y2']),
                        'score': current_row['score'] + ratio * (next_row['score'] - current_row['score']),
                        'frame_number': new_frame_number
                    }

                    # 将插值行添加到填充数据中
                    track_filled_df = track_filled_df.append(interpolated_row, ignore_index=True)

        # 添加最后一行
        track_filled_df = track_filled_df.append(track_df.iloc[-1], ignore_index=True)

        # 将当前 track_id 的填充数据添加到总的 DataFrame 中
        filled_df = filled_df.append(track_filled_df, ignore_index=True)

    # 对填充后的数据按照 track_id 和 frame_number 进行排序
    filled_df = filled_df.sort_values(['track_id', 'frame_number']).reset_index(drop=True)

    # 删除辅助的 frame_number 列
    filled_df = filled_df.drop(columns=['frame_number'])

    return filled_df


# 示例用法：
# 读取 Excel 文件
input_excel = '/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box/cloudy/26_06_2022_12_57_13_scene1_stop_cross_veryfar/detections.xlsx'  # 替换为您的 Excel 文件名
df = pd.read_excel(input_excel)

track_id_list = [1,2,10]  # 只处理 track_id 为 10 的数据
filled_df = fill_missing_frames(df, track_id_list=track_id_list)

# 调用函数，填充缺失帧
# filled_df = fill_missing_frames(df)

# 将填充后的数据保存到新的 Excel 文件
output_excel = '/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box/cloudy/01_06_2022_18_35_09_scene1_walk_cross_far/detections_过渡.xlsx'  # 输出文件名
filled_df.to_excel(output_excel, index=False)

print(f'填充后的数据已保存到 {output_excel}')
