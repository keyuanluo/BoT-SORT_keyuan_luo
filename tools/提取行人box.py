import cv2
import pandas as pd

# 读取原始图像
image = cv2.imread('image.jpg')

# 获取图像的尺寸
height, width, channels = image.shape

# 读取 Excel 文件，假设文件名为 'coordinates.xlsx'，并且数据在第一个工作表中
df = pd.read_excel('coordinates.xlsx')

# 假设 Excel 表格有以下列：'x1', 'y1', 'x2', 'y2'
boxes = df[['x1', 'y1', 'x2', 'y2']].values.tolist()

# 遍历每个 box 并提取对应的行人小图片
for idx, box in enumerate(boxes):
    x1, y1, x2, y2 = box

    # 边界检查，确保坐标在图像范围内
    x1 = int(max(0, min(x1, width - 1)))
    y1 = int(max(0, min(y1, height - 1)))
    x2 = int(max(0, min(x2, width - 1)))
    y2 = int(max(0, min(y2, height - 1)))

    # 如果 x2 小于等于 x1 或 y2 小于等于 y1，跳过此 box
    if x2 <= x1 or y2 <= y1:
        print(f"第 {idx} 个 box 的坐标无效，已跳过。")
        continue

    # 裁剪行人区域
    pedestrian_img = image[y1:y2, x1:x2]

    # 保存提取的行人小图片
    cv2.imwrite(f'pedestrian_{idx}.jpg', pedestrian_img)

    print(f"已保存第 {idx} 个行人图片：pedestrian_{idx}.jpg")
