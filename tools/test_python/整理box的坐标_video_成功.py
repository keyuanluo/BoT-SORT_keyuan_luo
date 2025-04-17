import os
import pandas as pd

def main():
    # 源文件夹，包含 video_0001 ... video_0255，每个下有 detections_最终.xlsx
    src_root = "/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box_video_number"
    # 目标文件夹（仅保存整理后的表格）
    dst_root = "/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box_video_number_02"

    # 若目标根目录不存在，先创建
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # 遍历 1..255
    for i in range(16,17):
        folder_name = f"video_{i:04d}"  # e.g. "video_0001"
        src_folder = os.path.join(src_root, folder_name)
        if not os.path.isdir(src_folder):
            continue

        # 源 Excel 文件
        src_excel = os.path.join(src_folder, "detections_最终.xlsx")
        if not os.path.isfile(src_excel):
            # 若不存在，则跳过
            continue

        # 1. 读取 Excel
        df = pd.read_excel(src_excel)

        # 2. 去除 image_name 列中的 ".png" 后缀，并改列名为 "image_number"
        if "image_name" in df.columns:
            df["image_name"] = df["image_name"].astype(str).str.replace(".png", "")

        else:
            print(f"[警告] {src_excel} 中没有 'image_name' 列，跳过此列处理。")


        # 3. 重命名 x1,y1,x2,y2 -> bbox_x1,bbox_y1,bbox_x2,bbox_y2
        rename_map = {
            "x1": "bbox_x1",
            "y1": "bbox_y1",
            "x2": "bbox_x2",
            "y2": "bbox_y2",
        }
        for old_col, new_col in rename_map.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
            else:
                # 如果原列不存在，就创建空列，防止后面索引时报错
                df[new_col] = ""

        # 4. 只保留想要的 5 列
        keep_cols = ["image_name", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
        df = df[keep_cols]

        # 5. 输出到目标目录下，以 "video_XXXX.xlsx" 命名
        dst_excel = os.path.join(dst_root, f"{folder_name}.xlsx")
        df.to_excel(dst_excel, index=False)

        print(f"[处理完成] {src_excel} -> {dst_excel}")

    print("所有处理已完成。")

if __name__ == "__main__":
    main()
