import cv2
import pandas as pd
import os
import sys

def extract_pedestrians_from_folder(image_folder, excel_file, output_folder):
    print(f"\n开始处理文件夹：{image_folder}")

    # 检查输入文件夹是否存在
    if not os.path.exists(image_folder):
        print(f"输入文件夹不存在：{image_folder}")
        return

    # 检查 Excel 文件是否存在
    if not os.path.exists(excel_file):
        print(f"未找到 Excel 文件：{excel_file}，无法继续处理。")
        return

    # 读取 Excel 文件
    try:
        df = pd.read_excel(excel_file)
    except Exception as e:
        print(f"读取 Excel 文件失败：{excel_file}，错误：{e}")
        return

    # 检查必要的列是否存在
    required_columns = {'image_name', 'x1', 'y1', 'x2', 'y2'}
    if not required_columns.issubset(df.columns):
        print(f"Excel 文件缺少必要的列 {required_columns}，无法继续处理。")
        return

    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_files:
        print(f"文件夹中没有找到图片文件。")
        return

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历每个图片文件
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"无法读取图片文件：{image_path}，跳过。")
            continue

        height, width = image.shape[:2]

        # 获取当前图片对应的检测框
        image_boxes = df[df['image_name'] == image_file][['x1', 'y1', 'x2', 'y2']].values

        if len(image_boxes) == 0:
            print(f"图片 {image_file} 没有对应的坐标信息，跳过。")
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
                print(f"图片 {image_file} 的第 {pedestrian_idx} 个 box 坐标无效，已跳过。")
                pedestrian_idx += 1
                continue

            # 裁剪行人区域
            pedestrian_img = image[y1:y2, x1:x2]

            # 获取原始图片文件名（不含扩展名）
            image_name_without_ext = os.path.splitext(image_file)[0]

            # 生成输出文件名
            output_filename = f"{image_name_without_ext}_pedestrian_{pedestrian_idx}.jpg"
            output_path = os.path.join(output_folder, output_filename)

            # 保存截取的行人小图片
            cv2.imwrite(output_path, pedestrian_img)

            print(f"已保存：{output_filename}")

            # 更新当前图片的行人索引
            pedestrian_idx += 1

    print(f"\n文件夹 {image_folder} 处理完成！")

if __name__ == "__main__":
    # 输入文件夹路径
    image_folder = '/media/robert/4TB-SSD/watchped_dataset - 副本/Images/evening/12_08_2022_18_00_44_scene5_walk_stop_far'
    excel_file = '/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box/evening/12_08_2022_18_00_44_scene5_walk_stop_far/detections_最终.xlsx'
    output_folder = '/media/robert/4TB-SSD/漏网之鱼/evening/12_08_2022_18_00_44_scene5_walk_stop_far'

    # 您也可以通过命令行参数传入
    if len(sys.argv) == 4:
        image_folder = sys.argv[1]
        excel_file = sys.argv[2]
        output_folder = sys.argv[3]
    elif len(sys.argv) != 1:
        print("用法：python script.py [输入图片文件夹路径] [Excel文件路径] [输出文件夹路径]")
        sys.exit(1)

    extract_pedestrians_from_folder(image_folder, excel_file, output_folder)
