import os
import shutil
import pandas as pd

def main():
    # 1. 配置基础路径
    excel_file = "/media/robert/4TB-SSD/watchped_dataset - 副本/video_address_new.xlsx"  # 存储 new_folder 和 original_path 的 Excel
    excel_speed_changed_root = "/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_车速_changed"  # 源头大文件夹，下面有 11 个子文件夹
    target_root = "/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_车速_video_numer"         # 目标文件夹，用于重命名后结果

    # 若目标根目录不存在，就创建
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    # 2. 读入 Excel (video_address_new.xlsx)
    df = pd.read_excel(excel_file)
    # 假设两列名称分别是 "new_folder" 和 "original_path"
    # - new_folder:  比如 "video_0001", "video_0002", ...
    # - original_path:  比如 "/media/.../Images/cloudy/26_06_2022_13_31_30_scene1_walk_cross_close"

    # 3. 遍历每行，定位要复制的 Excel 文件
    for idx, row in df.iterrows():
        new_folder_name = str(row["new_folder"]).strip()  # e.g. "video_0001"
        orig_path = str(row["original_path"]).strip()     # e.g. "/media/.../Images/cloudy/26_06_2022_13_31_30_scene1_walk_cross_close"

        # 解析最后一级目录
        # 示例：orig_path = "/media/.../Images/cloudy/26_06_2022_13_31_30_scene1_walk_cross_close"
        #   basename_1 = "26_06_2022_13_31_30_scene1_walk_cross_close"
        basename_1 = os.path.basename(orig_path)

        # 解析倒数第二级目录
        #  parent_dir = "/media/.../Images/cloudy"
        #  basename_2 = "cloudy"
        parent_dir = os.path.dirname(orig_path)
        basename_2 = os.path.basename(parent_dir)

        # 在 "Excel_车速_changed" 下的真实 Excel 文件名通常是 "basename_1 + .xlsx"
        # 并且位于子文件夹 basename_2 下
        # 组装出完整源文件路径
        source_excel = os.path.join(excel_speed_changed_root, basename_2, f"{basename_1}.xlsx")

        if not os.path.exists(source_excel):
            print(f"[Warning] 源 Excel 文件不存在: {source_excel}")
            continue

        # 目标文件名：例如 "video_0001.xlsx"
        new_excel_name = f"{new_folder_name}.xlsx"
        target_excel = os.path.join(target_root, new_excel_name)

        # 如果目标文件已存在，可以自行决定是否覆盖或跳过
        if os.path.exists(target_excel):
            print(f"[Info] 目标文件已存在，跳过: {target_excel}")
            continue

        # 4. 复制文件，并重命名
        print(f"复制 Excel:\n  源: {source_excel}\n  --> 目标: {target_excel}")
        shutil.copyfile(src=source_excel, dst=target_excel)

    print("\n所有 Excel 文件复制/重命名操作执行完毕。")


if __name__ == "__main__":
    main()
