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
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
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
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(t.cls)

                    # 保存结果
                    results.append(
                        f"{i + 1},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )

                    # 收集数据以导出到 Excel
                    detection_results.append({
                        'image_name': Path(p).name,
                      #  '间隔帧数':
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

    print(f'完成。({time.time() - t0:.3f}s)')


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
        weights='pretrained/yolov7-w6.pt',
        source='/media/robert/4TB-SSD/watchped_dataset - 副本/Images/cloudy/01_06_2022_18_24_49_scene1_stop_all',  # 修改为您的大文件夹路径
        img_size=1920,
        conf_thres=0.09,
        iou_thres=0.7,
        device='',  # 使用默认设备
        view_img=False,
        save_txt=False,
        save_conf=False,
        nosave=False,
        classes=[0],  # 仅跟踪类别为 0 的对象
        agnostic_nms=True,
        augment=False,
        update=False,
        project='/media/robert/4TB-SSD/box_excel',
        name='test',
        exist_ok=False,
        trace=False,
        hide_labels_name=False,
        # 跟踪参数
        track_high_thresh=0.3,
        track_low_thresh=0.05,
        new_track_thresh=0.4,
        track_buffer=30,
        match_thresh=0.75,
        aspect_ratio_thresh=1.6,
        min_box_area=10,
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

    print(opt)
    with torch.no_grad():
        main()
