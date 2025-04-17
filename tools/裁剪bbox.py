import os
import pandas as pd
from PIL import Image


def main():
    # 1. 配置基础路径
    excel_box_root = "/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box_video_number"  # 存放 video_0001...video_0255，每个下有 detections_最终.xlsx
    images_changed_root = "/media/robert/4TB-SSD/watchped_dataset - 副本/images_video_changed"  # 存放对应的原始图片
    output_root = "/media/robert/4TB-SSD/watchped_dataset - 副本/Box_video_new_01"  # 用于保存裁剪后的图片

    # 若目标根目录不存在，就创建
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 2. 遍历所有 video_0001 ... video_0255
    for i in range(16,17):
        video_name = f"video_{i:04d}"  # e.g. "video_0001"
        excel_folder = os.path.join(excel_box_root, video_name)
        excel_path = os.path.join(excel_folder, "detections_最终.xlsx")

        # 检查当前文件夹和Excel是否存在
        if not os.path.isdir(excel_folder):
            continue
        if not os.path.isfile(excel_path):
            continue

        # 3. 读取 detections_最终.xlsx
        df = pd.read_excel(excel_path)
        if df.empty:
            print(f"[Info] {video_name} 的 detections_最终.xlsx 为空，跳过处理。")
            continue

        # 4. 源图片所在目录
        image_folder = os.path.join(images_changed_root, video_name)
        if not os.path.isdir(image_folder):
            print(f"[Warning] 找不到源图像目录: {image_folder}，跳过 {video_name}")
            continue

        # 5. 为当前 video 创建输出目录
        output_folder = os.path.join(output_root, video_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 6. 遍历每行 BBox 并进行裁剪
        for idx, row in df.iterrows():
            image_name = str(row.get("image_name", "")).strip()  # e.g. "000343.png"
            x1 = row.get("x1", None)
            y1 = row.get("y1", None)
            x2 = row.get("x2", None)
            y2 = row.get("y2", None)

            # 若关键字段为空或非数值，则跳过
            if not image_name or pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2):
                continue

            # 拼出源图像路径
            src_path = os.path.join(image_folder, image_name)
            if not os.path.isfile(src_path):
                # 如果图片不存在，跳过
                # 也可打印提醒
                # print(f"[Warning] {video_name} 中找不到 {image_name}，跳过。")
                continue

            # 打开并裁剪
            try:
                with Image.open(src_path) as img:
                    # PIL 的 crop box 是 (left, upper, right, lower)
                    # x1,y1,x2,y2通常是 float，这里取 int 防止报错
                    crop_box = (int(x1), int(y1), int(x2), int(y2))
                    cropped_img = img.crop(crop_box)

                    # 生成输出文件名: 例如 "000343_pedestrian_1.jpg"
                    # 从原图像名里去掉扩展名，再加 "_pedestrian_1.jpg"
                    base_name = os.path.splitext(image_name)[0]  # e.g. "000343"
                    out_name = f"{base_name}_pedestrian_1.jpg"
                    out_path = os.path.join(output_folder, out_name)

                    # 保存裁剪图像
                    cropped_img.save(out_path, format="JPEG")  # 明确使用 JPG 格式
            except Exception as e:
                # 如果裁剪失败或者PIL异常，可在此打印
                print(f"[Error] 裁剪 {src_path} 时出错: {e}")

        print(f"[Info] 已处理完 {video_name} 的所有 BBox，并将结果保存在 {output_folder}")

    print("\n全部处理完成。")


if __name__ == "__main__":
    main()
