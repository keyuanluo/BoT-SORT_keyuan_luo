import time
from pathlib import Path
import sys
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import pandas as pd

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_imshow, non_max_suppression, \
    scale_coords, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, time_synchronized, TracedModel

from tracker.mc_bot_sort import BoTSORT

sys.path.insert(0, './yolov7')
sys.path.append('.')


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1),
                                          w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    print('保存结果到 {}'.format(filename))


def detect(source=None, save_dir=None):
    global opt  # 使用全局的 opt 变量
    if source is None:
        source = opt.source
    if save_dir is None:
        # 如果未提供 save_dir，则创建一个
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录

    save_img = not opt.nosave and not source.endswith('.txt')  # 保存推理图像
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # 初始化
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # 仅在 CUDA 上支持半精度

    # 加载模型
    weights = opt.weights
    imgsz = opt.img_size
    trace = opt.trace
    model = attempt_load(weights, map_location=device)  # 加载 FP32 模型
    stride = int(model.stride.max())  # 模型步长
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸

    if trace:
        model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # 转为 FP16

    # 二次分类器
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # 初始化
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # 设置数据加载器
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # 将 True 设置为加速恒定图像尺寸推理
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 获取名称和颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # 创建跟踪器
    tracker = BoTSORT(opt, frame_rate=30.0)

    # 初始化用于保存检测结果的列表
    detection_results = []
    last_frame_dict = {}  # 用于记录每个 track_id 上一次出现的帧号

    # 运行推理
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 运行一次
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 转 fp16/32
        img /= 255.0  # 0 - 255 转为 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # 应用 NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # 应用分类器
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # 处理检测结果
        results = []

        for i, det in enumerate(pred):  # 每张图像的检测
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # 从文件名中提取实际帧号
            try:
                frame_number = int(Path(p).stem)
            except ValueError:
                # 如果文件名不是数字，需要根据实际情况处理
                filename = Path(p).stem  # 获取文件名（不含扩展名）
                # 假设文件名格式为 'frame_000417'
                frame_number = int(''.join(filter(str.isdigit, filename)))  # 提取数字部分
            current_frame = frame_number  # 使用实际的帧号

            # 运行跟踪器
            detections = []
            if len(det):
                boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                boxes = boxes.cpu().numpy()
                detections = det.cpu().numpy()
                detections[:, :4] = boxes

            online_targets = tracker.update(detections, im0)

            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []
            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                tcls = t.cls

                # 计算间隔帧数
                if tid in last_frame_dict:
                    frame_gap = current_frame - last_frame_dict[tid]
                else:
                    frame_gap = 0  # 如果是首次出现，间隔帧数为0

                # 更新 last_frame_dict
                last_frame_dict[tid] = current_frame

                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(t.cls)

                    # 保存结果
                    results.append(
                        f"{current_frame},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )

                    # 收集数据以导出到 Excel
                    detection_results.append({
                        'image_name': Path(p).name,
                        'frame_gap': frame_gap,
                        'track_id': tid,
                        'class': names[int(tcls)],
                        'x1': tlbr[0],
                        'y1': tlbr[1],
                        'x2': tlbr[2],
                        'y2': tlbr[3],
                        'score': t.score
                    })

                    if save_img or opt.view_img:  # 在图像上绘制边界框
                        if opt.hide_labels_name:
                            label = f'{tid}, {int(tcls)}'
                        else:
                            label = f'{tid}, {names[int(tcls)]}'
                        plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
            p = Path(p)  # 转为 Path 对象
            save_path = str(save_dir / p.name)  # img.jpg

            # 打印时间（推理 + NMS）
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # 显示结果
            if opt.view_img:
                cv2.imshow('BoT-SORT', im0)
                cv2.waitKey(1)  # 1 毫秒

            # 保存结果（带检测的图像）
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # 新视频
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # 释放之前的视频写入器
                        if vid_cap:  # 视频
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # 流
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    # 导出检测结果到 Excel
    if detection_results:
        df = pd.DataFrame(detection_results)
        excel_save_path = save_dir / 'detections.xlsx'
        df.to_excel(excel_save_path, index=False)
        print(f'检测结果已保存到 {excel_save_path}')

        # # 调用插值函数，对指定的 track_id 进行缺失帧填充
        # track_id_list = [10]  # 指定需要填充的 track_id 列表
        # filled_df = fill_missing_frames(df, track_id_list=track_id_list)
        filled_df = fill_missing_frames(df)

        # 将填充后的数据保存到新的 Excel 文件
        filled_excel_save_path = save_dir / 'detections_filled.xlsx'
        filled_df.to_excel(filled_excel_save_path, index=False)
        print(f'填充后的数据已保存到 {filled_excel_save_path}')

    print(f'完成。({time.time() - t0:.3f}s)')


def fill_missing_frames(df, track_id_list=None):
    # 提取帧号，假设 image_name 的格式为 '000370.png'
    df['frame_number'] = df['image_name'].str.extract('(\d+)').astype(int)

    # 获取所有的 track_id
    all_track_ids = df['track_id'].unique()

    # 用于存储填充后的所有数据
    filled_df = pd.DataFrame()

    for track_id in all_track_ids:
        # 筛选出当前 track_id 的数据，并按照帧号排序
        track_df = df[df['track_id'] == track_id].sort_values('frame_number').reset_index(drop=True)

        # 创建一个新的 DataFrame 来存储当前 track_id 的填充数据
        track_filled_df = pd.DataFrame()

        # 判断当前的 track_id 是否在指定的 track_id_list 中
        if track_id_list is None or track_id in track_id_list:
            # 对指定的 track_id 进行缺失帧填充
            for idx in range(len(track_df) - 1):
                current_row = track_df.loc[idx]
                next_row = track_df.loc[idx + 1]

                # 计算当前帧和下一帧之间的差值
                frame_diff = next_row['frame_number'] - current_row['frame_number']

                # 将当前行添加到填充数据中
                track_filled_df = track_filled_df.append(current_row, ignore_index=True)

                # 如果帧差值大于1，表示存在缺失帧，需要进行插值
                if frame_diff > 1:
                    for gap in range(1, frame_diff):
                        # 计算插值比例
                        ratio = gap / frame_diff

                        # 生成新的帧号
                        new_frame_number = current_row['frame_number'] + gap
                        new_image_name = '{:06d}.png'.format(new_frame_number)

                        # 对 x1, y1, x2, y2, score 进行线性插值
                        interpolated_row = {
                            'image_name': new_image_name,
                            'frame_gap': 1,
                            'track_id': track_id,
                            'class': current_row['class'],
                            'x1': current_row['x1'] + ratio * (next_row['x1'] - current_row['x1']),
                            'y1': current_row['y1'] + ratio * (next_row['y1'] - current_row['y1']),
                            'x2': current_row['x2'] + ratio * (next_row['x2'] - current_row['x2']),
                            'y2': current_row['y2'] + ratio * (next_row['y2'] - current_row['y2']),
                            'score': current_row['score'] + ratio * (next_row['score'] - current_row['score']),
                            'frame_number': new_frame_number
                        }

                        # 将插值行添加到填充数据中
                        track_filled_df = track_filled_df.append(interpolated_row, ignore_index=True)
            # 添加最后一行
            track_filled_df = track_filled_df.append(track_df.iloc[-1], ignore_index=True)
        else:
            # 对于未指定的 track_id，直接添加原始数据
            track_filled_df = track_df.copy()

        # 将当前 track_id 的填充数据添加到总的 DataFrame 中
        filled_df = filled_df.append(track_filled_df, ignore_index=True)

    # 对填充后的数据按照 track_id 和 frame_number 进行排序
    filled_df = filled_df.sort_values(['track_id', 'frame_number']).reset_index(drop=True)

    # 删除辅助的 frame_number 列
    filled_df = filled_df.drop(columns=['frame_number'])

    return filled_df


def main():
    source = opt.source
    if os.path.isdir(source):
        subdirs = [os.path.join(source, d) for d in os.listdir(source)
                   if os.path.isdir(os.path.join(source, d))]
        if subdirs:
            for subdir in subdirs:
                source_subdir = subdir
                # 为每个子文件夹创建单独的保存目录
                save_dir = Path(opt.project) / opt.name / os.path.basename(subdir)
                save_dir = Path(increment_path(save_dir, exist_ok=opt.exist_ok))
                (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)
                # 调用 detect()
                detect(source=source_subdir, save_dir=save_dir)
        else:
            # 如果没有子文件夹，则处理源目录
            save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
            (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)
            detect(source=source, save_dir=save_dir)
    else:
        # 如果 source 不是目录，直接处理
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)
        detect(source=source, save_dir=save_dir)


if __name__ == '__main__':
    from argparse import Namespace

    opt = Namespace(
        weights='pretrained/yolov7-e6e.pt',
        source='/media/robert/4TB-SSD/watchped_dataset - 副本/Images/sunny 2',  # 修改为您的大文件夹路径
        #img_size=1920,
        img_size=2560,
        #img_size=3840,
        conf_thres=0.03,
        iou_thres=0.7,
        # iou_thres=0.9,
        device='',  # 使用默认设备
        view_img=False,
        save_txt=False,
        save_conf=False,
        nosave=False,
        classes=[0],  # 仅跟踪类别为 0 的对象
        agnostic_nms=True,
        #augment=False,
        augment=True,
        update=False,
        project='/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box',
        name='',
        exist_ok=False,
        trace=False,
        hide_labels_name=False,
        # 跟踪参数
        track_high_thresh=0.3,
        track_low_thresh=0.05,
        new_track_thresh=0.4,
        # track_high_thresh=0.1,
        # track_low_thresh=0.01,
        # new_track_thresh=0.2,
        track_buffer=5,  #像素
        match_thresh=0.75,
        aspect_ratio_thresh=1.6,
        min_box_area=1,
        mot20=True,  # 对应 --fuse-score
        # CMC
        cmc_method='sparseOptFlow',
        # ReID
        with_reid=True,  # 如果不使用 ReID 模块，可以设置为 False
        fast_reid_config='fast_reid/configs/MOT20/sbs_S50.yml',
        fast_reid_weights='pretrained/mot20_sbs_S50.pth',
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        # 其他选项
        jde=False,
        ablation=False,
    )
    # 动态设置 opt.name，与 opt.source 的最后一个目录名一致
    opt.name = os.path.basename(opt.source.rstrip('/\\'))
    print(opt)
    with torch.no_grad():
        main()
