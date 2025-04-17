import os
import shutil
import pandas as pd

def main():
    # 1. 配置基础路径
    excel_file = "/media/robert/4TB-SSD/watchped_dataset - 副本/video_address_new.xlsx"  # Excel 文件
    excel_box_root = "/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box"          # 源头大文件夹
    target_root = "/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box_video_number" # 目标文件夹，用于存放重命名后结果

    # 若目标根目录不存在，就创建
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    # 2. 读入 Excel
    df = pd.read_excel(excel_file)
    # 假设两列名称分别是 "new_folder" 和 "original_path"
    # new_folder:  比如 video_0001, video_0002, ...
    # original_path:  比如 /media/.../Images/evening 2/01_06_2023_17_07_54_scene7_stop_cross_veryfar

    # 3. 遍历每行，定位要复制的子子文件夹
    for idx, row in df.iterrows():
        new_folder_name = str(row["new_folder"]).strip()   # e.g. "video_0001"
        orig_path = str(row["original_path"]).strip()      # e.g. "/media/.../Images/evening 2/01_06_2023_17_07_54_scene7_stop_cross_veryfar"

        # 由于 excel 里存的是 Images 路径，但实际要从 Excel_Box 找到对应的 "子文件夹/子子文件夹"，
        # 我们可以解析出最后两级目录，然后拼到 Excel_Box 里去。
        # 比如：
        #   orig_path = .../Images/evening 2/01_06_2023_17_07_54_scene7_stop_cross_veryfar
        #   basename_1 = "01_06_2023_17_07_54_scene7_stop_cross_veryfar"  (最后一级)
        #   basename_2 = "evening 2"  (倒数第二级)
        #   source_dir = /media/robert/4TB-SSD/.../Excel_Box/evening 2/01_06_2023_17_07_54_scene7_stop_cross_veryfar

        # 解析最后一级子目录名
        basename_1 = os.path.basename(orig_path)  # e.g. "01_06_2023_17_07_54_scene7_stop_cross_veryfar"
        # 解析倒数第二级子目录名
        parent_dir = os.path.dirname(orig_path)   # e.g. ".../Images/evening 2"
        basename_2 = os.path.basename(parent_dir) # e.g. "evening 2"

        # 拼出在 Excel_Box 下的真实完整路径
        source_dir = os.path.join(excel_box_root, basename_2, basename_1)
        if not os.path.exists(source_dir):
            print(f"[Warning] 源路径不存在: {source_dir}. 跳过处理。")
            continue

        # 目标路径：Excel_Box_video_number/video_xxxx
        target_dir = os.path.join(target_root, new_folder_name)

        # 如果目标路径已存在，可能需自行决定如何处理：是覆盖、跳过、还是先删除？
        if os.path.exists(target_dir):
            print(f"[Info] 目标路径已存在，可能已复制过，跳过: {target_dir}")
            continue

        # 4. 复制整个文件夹
        print(f"复制:\n  源: {source_dir}\n  --> 目标: {target_dir}")
        shutil.copytree(src=source_dir, dst=target_dir)

    print("\n所有复制操作执行完毕。")


if __name__ == "__main__":
    main()
