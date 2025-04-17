import os
import pandas as pd


def main():
    # 根目录，包含 video_0001 ... video_0255 子文件夹
    root_dir = "/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box_video_number"

    # 遍历所有 video 文件夹（从 1 ~ 255）
    for i in range(1, 256):
        video_folder = f"video_{i:04d}"  # e.g. "video_0001"
        video_path = os.path.join(root_dir, video_folder)
        if not os.path.isdir(video_path):
            # 如果该文件夹不存在，跳过
            continue

        # 查找子文件夹中的 detections_最终.xlsx
        excel_path = os.path.join(video_path, "detections_最终.xlsx")
        if not os.path.isfile(excel_path):
            # 没有这个文件则跳过
            continue

        # 读入 Excel
        df = pd.read_excel(excel_path)

        # ---------- 检查 1: x2>=x1 & y2>=y1 ----------
        invalid_mask = (df["x2"] < df["x1"]) | (df["y2"] < df["y1"])
        invalid_rows = df[invalid_mask]
        if not invalid_rows.empty:
            print(f"[Check1] 警告: {video_folder} 中下列帧不满足 (x2>=x1, y2>=y1):")
            frames = invalid_rows["image_name"].unique()
            print("  重叠或反向 BBox 的帧:", list(frames))

        # ---------- 检查 2: 重复帧 --------------
        # 假设用 'image_name' 来判断帧是否重复（如需考虑 track_id，则改为 ['track_id','image_name']）
        dup_mask = df.duplicated(subset=["image_name"], keep=False)
        dup_df = df[dup_mask]
        if not dup_df.empty:
            print(f"[Check2] 警告: {video_folder} 中下列帧重复出现:")
            dup_frames = dup_df["image_name"].unique()
            for f in dup_frames:
                print(f"    重复帧: {f}")

            # 只保留首次出现的行
            df_dedup = df.drop_duplicates(subset=["image_name"], keep="first")

            # 将去重结果另存为 detections_最终_去重.xlsx（不覆盖原文件，方便对比）
            out_path = os.path.join(video_path, "detections_最终_1111.xlsx")
            df_dedup.to_excel(out_path, index=False)
            print(f"    已去重并另存为: {out_path}")
        else:
            # 无重复帧
            pass


if __name__ == "__main__":
    main()
