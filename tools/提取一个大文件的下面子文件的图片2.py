import cv2
import pandas as pd
import os
import sys

def extract_pedestrians_from_folders(input_big_folder, output_big_folder):
    # 检查输入文件夹是否存在
    if not os.path.exists(input_big_folder):
        print(f"输入文件夹不存在：{input_big_folder}")
        sys.exit(1)

    # 创建输出大文件夹，如果不存在
    os.makedirs(output_big_folder, exist_ok=True)

    # 遍历输入大文件夹中的所有子文件夹
    for subfolder in os.listdir(input_big_folder):
        subfolder_path = os.path.join(input_big_folder, subfolder)

        if os.path.isdir(subfolder_path):
            print(f"\n处理子文件夹：{subfolder}")

            # 查找 Excel 文件
            excel_file = os.path.join(subfolder_path, 'detections_最终.xlsx')
            if not os.path.exists(excel_file):
                print(f"  未找到 Excel 文件：{excel_file}，跳过该子文件夹。")
                continue

            # 读取 Excel 文件
            try:
                df = pd.read_excel(excel_file)
            except Exception as e:
                print(f"  读取 Excel 文件失败：{excel_file}，错误：{e}")
                continue

            # 检查必要的列是否存在
            required_columns = {'image_name', 'x1', 'y1', 'x2', 'y2'}
            if not required_columns.issubset(df.columns):
                print(f"  Excel 文件缺少必要的列 {required_columns}，跳过该子文件夹。")
                continue

            # 获取子文件夹中的所有图片文件
            image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            if not image_files:
                print(f"  子文件夹中没有找到图片文件，跳过。")
                continue

            # 创建对应的输出子文件夹
            output_subfolder = os.path.join(output_big_folder, subfolder)
            os.makedirs(output_subfolder, exist_ok=True)

            # 遍历每个图片文件
            for image_file in image_files:
                image_path = os.path.join(subfolder_path, image_file)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"  无法读取图片文件：{image_path}，跳过。")
                    continue

                height, width, channels = image.shape

                # 获取当前图片对应的检测框
                image_boxes = df[df['image_name'] == image_file][['x1', 'y1', 'x2', 'y2']].values

                if len(image_boxes) == 0:
                    print(f"  图片 {image_file} 没有对应的坐标信息，跳过。")
                    continue

                # 初始化当前图片的行人索引
                pedestrian_idx = 1

                # 遍历每个检测框并提取行人小图片
                for (x1, y1, x2, y2) in image_boxes:
                    # 边界检查
                    x1 = int(max(0, min(x1, width - 1)))
                    y1 = int(max(0, min(y1, height - 1)))
                    x2 = int(max(0, min(x2, width - 1)))
                    y2 = int(max(0, min(y2, height - 1)))

                    # 坐标有效性检查
                    if x2 <= x1 or y2 <= y1:
                        print(f"    图片 {image_file} 的第 {pedestrian_idx} 个 box 坐标无效，已跳过。")
                        pedestrian_idx += 1
                        continue

                    # 裁剪行人区域
                    pedestrian_img = image[y1:y2, x1:x2]

                    # 获取原始图片文件名（不含扩展名）
                    image_name_without_ext = os.path.splitext(image_file)[0]

                    # 生成输出文件名
                    output_filename = f"{image_name_without_ext}_pedestrian_{pedestrian_idx}.jpg"
                    output_path = os.path.join(output_subfolder, output_filename)

                    # 保存截取的行人小图片
                    cv2.imwrite(output_path, pedestrian_img)

                    print(f"    已保存：{output_filename}")

                    # 更新当前图片的行人索引
                    pedestrian_idx += 1

    print("\n所有处理完成！")

if __name__ == "__main__":
    # 示例输入输出路径，可以根据需要修改或通过命令行参数传入
    input_big_folder = '/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box/cloudy'   # 请将此路径修改为您的输入大文件夹路径
    output_big_folder = '/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box_Walker/cloudy'    # 请将此路径修改为您的输出文件夹路径

    # 您也可以通过命令行参数传入路径
    if len(sys.argv) == 3:
        input_big_folder = sys.argv[1]
        output_big_folder = sys.argv[2]
    elif len(sys.argv) != 1:
        print("用法：python script.py [输入大文件夹路径] [输出大文件夹路径]")
        sys.exit(1)

    extract_pedestrians_from_folders(input_big_folder, output_big_folder)
