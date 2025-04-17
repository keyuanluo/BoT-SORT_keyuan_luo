import argparse
import time
from pathlib import Path
import sys

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolov11.models.experimental import attempt_load
from yolov11.utils.datasets import LoadStreams, LoadImages
from yolov11.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov11.utils.plots import plot_one_box
from yolov11.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

sys.path.insert(0, './yolov11')
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
    print('结果已保存到 {}'.format(filename))


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    save_img = not opt.nosave and not source.endswith('.txt')  # 保存推理图像
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # 目录设置
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # 增量运行
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录

    # 初始化
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # 只有CUDA支持半精度

    # 加载模型
    model = attempt_load(weights, map_location=device)  # 加载FP32模型
    stride = int(model.stride.max())  # 模型步长
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像尺寸

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # 转为FP16

    # 第二阶段分类器
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # 初始化
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # 设置数据加载器
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # 设置为True以加速常尺寸推理
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 获取类别名称和颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # 创建跟踪器
    tracker = BoTSORT(opt, frame_rate=30.0)

    # 开始推理
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 预热一次
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8转fp16/32
        img /= 255.0  # 0 - 255 转为 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # 应用NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # 应用分类器
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # 处理检测结果
        results = []

        for i, det in enumerate(pred):  # 每张图像的检测结果
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

                    if save_img or view_img:  # 在图像上添加边框
                        if opt.hide_labels_name:
                            label = f'{tid}, {int(tcls)}'
                        else:
                            label = f'{tid}, {names[int(tcls)]}'
                        plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
            p = Path(p)  # 转为Path对象
            save_path = str(save_dir / p.name)  # img.jpg

            # 打印时间（推理 + NMS）
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # 显示结果
            if view_img:
                cv2.imshow('BoT-SORT', im0)
                cv2.waitKey(1)  # 1毫秒

            # 保存结果（带检测的图像）
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' 或 'stream'
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

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} 标签已保存到 {save_dir / 'labels'}" if save_txt else ''
        # print(f"结果已保存到 {save_dir}{s}")

    print(f'完成. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov11.pt', help='模型路径(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='数据源')  # 文件/文件夹，0表示摄像头
    parser.add_argument('--img-size', type=int, default=1920, help='推理尺寸（像素）')
    parser.add_argument('--conf-thres', type=float, default=0.09, help='目标置信度阈值')  # 默认值为0.09
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS的IOU阈值')  # 默认值为0.7
    parser.add_argument('--device', default='', help='CUDA设备，例如0或0,1,2,3或cpu')
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    parser.add_argument('--save-txt', action='store_true', help='将结果保存到*.txt')
    parser.add_argument('--save-conf', action='store_true', help='在--save-txt标签中保存置信度')
    parser.add_argument('--nosave', action='store_true', help='不保存图像/视频')
    parser.add_argument('--classes', nargs='+', type=int, help='按类别过滤：--class 0，或 --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='类别无关的NMS')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--update', action='store_true', help='更新所有模型')
    parser.add_argument('--project', default='runs/detect', help='将结果保存到project/name')
    parser.add_argument('--name', default='exp', help='将结果保存到project/name')
    parser.add_argument('--exist-ok', action='store_true', help='允许已存在的project/name，不递增')
    parser.add_argument('--trace', action='store_true', help='跟踪模型')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='隐藏标签')

    # 跟踪参数
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="跟踪置信度阈值")  # 默认值是0.3
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="最低检测阈值")  # 默认值是0.05
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="新轨迹阈值")  # 默认值为0.4
    parser.add_argument("--track_buffer", type=int, default=30, help="保持丢失轨迹的帧数")  # 默认值为30
    parser.add_argument("--match_thresh", type=float, default=0.75, help="跟踪匹配阈值")  # 默认值为0.75
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="过滤出长宽比超过给定值的框的阈值。")  # 默认值为1.6
    parser.add_argument('--min_box_area', type=float, default=10, help='过滤出极小的框')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="融合得分和IOU进行关联")  # mot20不能更改

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str,
                        help="CMC方法：sparseOptFlow | files (Vidstab GMC) | orb | ecc")
    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="使用ReID模块。")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config",
                        default=r"fast_reid/configs/MOT20/sbs_S50.yml",
                        type=str, help="ReID配置文件路径")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot20_sbs_S50.pth",
                        type=str, help="ReID权重文件路径")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='用于拒绝低重叠ReID匹配的阈值')  # 默认值0.5
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='用于拒绝低外观相似性ReID匹配的阈值')  # 默认值0.25

    opt = parser.parse_args()
    opt.classes = [0]  # 此模型仅跟踪行人

    opt.jde = False
    opt.ablation = False

    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # 更新所有模型（修复SourceChangeWarning）
            for opt.weights in ['yolov11.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
