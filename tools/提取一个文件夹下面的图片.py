import cv2
import pandas as pd
import os

# 指定图片文件夹路径
image_folder = '//media/robert/4TB-SSD/watchped_dataset - 副本/Images/sunny 2/01_14_2023_12_05_13_scene8_stop_cross_close'  # 请将此路径修改为您的图片文件夹路径

# 读取Excel文件，假设文件名为 'coordinates.xlsx'
df = pd.read_excel('/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box/sunny 2/01_14_2023_12_05_13_scene8_stop_cross_close/detections_最终.xlsx')

# 假设Excel表格有以下列：'image_name', 'x1', 'y1', 'x2', 'y2'
# 确保 'image_name' 列中包含图片的文件名

# 获取文件夹下的所有图片文件
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# 创建输出文件夹
output_folder = '/media/robert/4TB-SSD/漏网之鱼/sunny 2'
os.makedirs(output_folder, exist_ok=True)

# 遍历每个图片文件
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图片文件：{image_path}")
        continue

    height, width, channels = image.shape

    # 获取当前图片对应的box坐标
    image_boxes = df[df['image_name'] == image_file][['x1', 'y1', 'x2', 'y2']].values

    if len(image_boxes) == 0:
        print(f"图片 {image_file} 没有对应的坐标信息，已跳过。")
        continue

    # 遍历每个box并提取行人小图片
    for idx, (x1, y1, x2, y2) in enumerate(image_boxes):
        # 边界检查
        x1 = int(max(0, min(x1, width - 1)))
        y1 = int(max(0, min(y1, height - 1)))
        x2 = int(max(0, min(x2, width - 1)))
        y2 = int(max(0, min(y2, height - 1)))

        # 坐标有效性检查
        if x2 <= x1 or y2 <= y1:
            print(f"图片 {image_file} 的第 {idx} 个box坐标无效，已跳过。")
            continue

        # 裁剪行人区域
        pedestrian_img = image[y1:y2, x1:x2]

        # 生成输出文件名
        output_filename = f"{os.path.splitext(image_file)[0]}_pedestrian_{idx}.jpg"
        output_path = os.path.join(output_folder, output_filename)

        # 保存提取的行人小图片
        cv2.imwrite(output_path, pedestrian_img)

        print(f"已保存图片 {image_file} 的第 {idx} 个行人图片：{output_filename}")
