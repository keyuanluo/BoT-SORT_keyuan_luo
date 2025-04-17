import cv2
import os
import csv
import pandas as pd

# 设置要标记的图像文件夹路径
image_folder = '/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box/sunny 2/01_14_2023_12_05_13_scene8_stop_cross_close'  # 修改为您的图像文件夹路径

# 让 output_csv 与输入文件夹相同
output_csv = os.path.join(image_folder, 'points_coordinates.csv')  # 输出的 CSV 文件路径

# 重塑后的 CSV 文件路径
reshaped_csv = os.path.join(image_folder, 'reshaped_points_coordinates.csv')

# # 如果存在旧的 CSV 文件，删除它
# if os.path.exists(output_csv):
#     os.remove(output_csv)

# 创建一个空的列表来存储标注结果
annotations = []

# 定义鼠标点击事件的回调函数
def mouse_callback(event, x, y, flags, param):
    global annotations, image_name, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'在图像 {image_name} 中点击了位置 ({x}, {y})')
        annotations.append({'image_name': image_name, 'x': x, 'y': y})
        # 在图像上绘制一个更小的红色圆点表示点击位置
        cv2.circle(img_display, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow(window_name, img_display)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右键点击，删除距离当前位置最近的标记点
        current_annotations = [ann for ann in annotations if ann['image_name'] == image_name]
        if current_annotations:
            # 计算所有标记点到点击位置的距离
            distances = [(idx, (ann['x'] - x) ** 2 + (ann['y'] - y) ** 2) for idx, ann in enumerate(current_annotations)]
            # 找到距离最近的标记点
            min_idx, min_dist = min(distances, key=lambda t: t[1])
            # 从 annotations 中删除该标记点
            removed_annotation = current_annotations[min_idx]
            annotations.remove(removed_annotation)
            print(f'删除了图像 {image_name} 中位置 ({removed_annotation["x"]}, {removed_annotation["y"]}) 的标记点')
            # 重新绘制图像
            img_display = img.copy()
            cv2.putText(img_display, image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # 重新绘制剩余的标记点
            for ann in annotations:
                if ann['image_name'] == image_name:
                    cv2.circle(img_display, (ann['x'], ann['y']), 3, (0, 0, 255), -1)
            cv2.imshow(window_name, img_display)
        else:
            print('当前图像没有可删除的标记点。')

# 获取图像文件列表
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
image_files.sort()  # 对文件名进行排序

# 创建窗口并设置为全屏
window_name = 'Image'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(window_name, mouse_callback)

is_fullscreen = True  # 全屏状态标志

current_image_index = 0  # 当前图像的索引
should_exit = False  # 标志位，控制程序是否退出

while 0 <= current_image_index < len(image_files):
    image_file = image_files[current_image_index]
    image_path = os.path.join(image_folder, image_file)
    image_name = image_file
    img = cv2.imread(image_path)
    if img is None:
        print(f'无法读取图像：{image_path}')
        current_image_index += 1
        continue

    # 显示图像名称
    img_display = img.copy()
    cv2.putText(img_display, image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 重新绘制已标注的点
    for annotation in annotations:
        if annotation['image_name'] == image_name:
            cv2.circle(img_display, (annotation['x'], annotation['y']), 3, (0, 0, 255), -1)

    print(f'正在标记图像：{image_name}')
    print('请在图像上点击鼠标左键以记录多个点的坐标。')
    print('右键点击删除最近的标记点。')
    print('按 "d" 键或右箭头键跳到下一张图像，按 "a" 键或左箭头键返回上一张图像，按 "s" 键保存标注结果，按 "m" 键切换全屏/窗口模式，按 "Esc" 键退出程序。')

    while True:
        # 显示图像
        cv2.imshow(window_name, img_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('d') or key == 83:  # 按 'd' 键或右箭头键跳到下一张图像
            if current_image_index < len(image_files) - 1:
                current_image_index += 1
            else:
                print("已经是最后一张图像，无法前进。")
            break
        elif key == ord('a') or key == 81:  # 按 'a' 键或左箭头键返回上一张图像
            if current_image_index > 0:
                current_image_index -= 1
            else:
                print("已经是第一张图像，无法返回上一张。")
            break
        elif key == ord('m'):  # 按 'm' 键切换全屏/窗口模式
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                is_fullscreen = False
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                is_fullscreen = True
        elif key == 27:  # 按 'Esc' 键退出程序
            should_exit = True
            break  # 跳出内层循环
        elif key == ord('r'):  # 按 'r' 键撤销最后一个标记点
            # 撤销当前图像的最后一个标记点
            current_annotations = [ann for ann in annotations if ann['image_name'] == image_name]
            if current_annotations:
                last_annotation = current_annotations[-1]
                annotations.remove(last_annotation)
                print(f'撤销了图像 {image_name} 中位置 ({last_annotation["x"]}, {last_annotation["y"]}) 的标记点')
                # 重新绘制图像
                img_display = img.copy()
                cv2.putText(img_display, image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                for ann in annotations:
                    if ann['image_name'] == image_name:
                        cv2.circle(img_display, (ann['x'], ann['y']), 3, (0, 0, 255), -1)
                cv2.imshow(window_name, img_display)
            else:
                print('当前图像没有可撤销的标记点。')
        elif key == ord('s'):  # 按 's' 键保存标注结果
            # 将标注结果保存到 CSV 文件
            with open(output_csv, 'w', newline='') as csvfile:
                fieldnames = ['image_name', 'x', 'y']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for annotation in annotations:
                    writer.writerow(annotation)
            print(f'标注结果已保存到 {output_csv}')
            # # 立即执行数据重塑
            # df = pd.read_csv(output_csv)
            # df['point_index'] = df.groupby('image_name').cumcount() + 1
            # df_wide = df.pivot(index='image_name', columns='point_index', values=['x', 'y'])
            # df_wide.columns = [f'{coord}{idx}' for coord, idx in df_wide.columns]
            # df_wide.fillna('', inplace=True)
            # df_wide.reset_index(inplace=True)
            # df_wide.to_csv(reshaped_csv, index=False)
            # print(f'数据已转换并保存到 {reshaped_csv}')
            # 执行数据重塑
            df = pd.read_csv(output_csv)
            df['point_index'] = df.groupby('image_name').cumcount() + 1

            # 重塑数据，并调整列顺序为 x1, y1, x2, y2, ...
            df_wide = df.pivot(index='image_name', columns='point_index', values=['x', 'y'])
            df_wide = df_wide.swaplevel(axis=1).sort_index(axis=1, level=0)  # 调整列顺序为 x1, y1, x2, y2
            df_wide.columns = [f'{coord}{idx}' for idx, coord in df_wide.columns]
            df_wide.fillna('', inplace=True)
            df_wide.reset_index(inplace=True)

            # 保存到 CSV 文件
            df_wide.to_csv(reshaped_csv, index=False)
            print(f'数据已转换并保存到 {reshaped_csv}')

    if should_exit:
        break  # 跳出外层循环

# 关闭所有窗口
cv2.destroyAllWindows()

# 程序结束前保存标注结果
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['image_name', 'x', 'y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for annotation in annotations:
        writer.writerow(annotation)
print(f'标注完成，结果已保存到 {output_csv}')

# 执行数据重塑
df = pd.read_csv(output_csv)
df['point_index'] = df.groupby('image_name').cumcount() + 1
df_wide = df.pivot(index='image_name', columns='point_index', values=['x', 'y'])
df_wide.columns = [f'{coord}{idx}' for coord, idx in df_wide.columns]
df_wide.fillna('', inplace=True)
df_wide.reset_index(inplace=True)
df_wide.to_csv(reshaped_csv, index=False)
print(f'数据已转换并保存到 {reshaped_csv}')
