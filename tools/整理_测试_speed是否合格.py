import os
import re
import pandas as pd

def main():
    # Excel 文件所在目录
    excel_dir = "/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_车速_video_numer"

    # 要检查的视频范围 (1..255)
    start_idx, end_idx = 1, 255

    # 定义一个正则，用于匹配“整数.小数” 格式
    # 例如 "10.5", "123.00", "0.123" 等都符合；但 "10" 或 "abc" 不符合
    pattern = re.compile(r'^\d+\.\d+$')

    # 用于记录是否出现了不合格
    any_invalid = False

    for i in range(start_idx, end_idx + 1):
        video_name = f"video_{i:04d}"
        excel_path = os.path.join(excel_dir, f"{video_name}.xlsx")

        if not os.path.isfile(excel_path):
            # 没有这个 Excel，则跳过
            continue

        df = pd.read_excel(excel_path)

        if "speed" not in df.columns:
            # 如果没有 speed 列，也可以视需要决定是否算作不合格
            # 此处仅提示并继续
            print(f"[警告] {excel_path} 中没有 speed 列，跳过检查。")
            continue

        for idx, row in df.iterrows():
            speed_value = row["speed"]
            # 转字符串以正则匹配
            speed_str = str(speed_value).strip()

            # 若不匹配则判定不合格
            if not pattern.match(speed_str):
                frame_id = row.get("image_name", f"index_{idx}")
                print(f"[不合格] {video_name}, 帧={frame_id}, speed={speed_str}")
                any_invalid = True

    # 如果遍历完所有 Excel 都没有出现不合格记录，则打印“全部合格”
    if not any_invalid:
        print("全部合格")

if __name__ == "__main__":
    main()
