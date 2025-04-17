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
    global opt  # Use the global opt variable
    if source is None:
        source = opt.source
    if save_dir is None:
        # Create one if save_dir is not provided
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # Create directory

    save_img = not opt.nosave and not source.endswith('.txt')  # Save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialization
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # Half precision is supported only on CUDA

    # Load model
    weights = opt.weights
    imgsz = opt.img_size
    trace = opt.trace
    model = attempt_load(weights, map_location=device)  # Load FP32 model
    stride = int(model.stride.max())  # Model stride
    imgsz = check_img_size(imgsz, s=stride)  # Check image size

    if trace:
        model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # Convert to FP16

    # Secondary classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # Initialization
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # Set True to speed up constant image-size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Create tracker
    tracker = BoTSORT(opt, frame_rate=30.0)

    # Initialize list to save detection results
    detection_results = []
    last_frame_dict = {}  # Record the last frame in which each track_id appeared

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # Run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        results = []

        for i, det in enumerate(pred):  # Detection for each image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # Extract actual frame number from filename
            try:
                frame_number = int(Path(p).stem)
            except ValueError:
                # If filename is not numeric, handle accordingly
                filename = Path(p).stem  # Get filename without extension
                # Assume filename format is 'frame_000417'
                frame_number = int(''.join(filter(str.isdigit, filename)))  # Extract numeric part
            current_frame = frame_number  # Use the actual frame number

            # Run tracker
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

                # Calculate gap between frames
                if tid in last_frame_dict:
                    frame_gap = current_frame - last_frame_dict[tid]
                else:
                    frame_gap = 0  # First appearance, gap is 0

                # Update last_frame_dict
                last_frame_dict[tid] = current_frame

                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(t.cls)

                    # Save results
                    results.append(
                        f"{current_frame},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )

                    # Collect data for exporting to Excel
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

                    if save_img or opt.view_img:  # Draw bounding box on image
                        if opt.hide_labels_name:
                            label = f'{tid}, {int(tcls)}'
                        else:
                            label = f'{tid}, {names[int(tcls)]}'
                        plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
            p = Path(p)  # Convert to Path object
            save_path = str(save_dir / p.name)  # img.jpg

            # Print timing (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Display results
            if opt.view_img:
                cv2.imshow('BoT-SORT', im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # New video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # Release previous video writer
                        if vid_cap:  # Video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # Stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    # Export detection results to Excel
    if detection_results:
        df = pd.DataFrame(detection_results)
        excel_save_path = save_dir / 'detections.xlsx'
        df.to_excel(excel_save_path, index=False)
        print(f'检测结果已保存到 {excel_save_path}')

        # # Call interpolation function to fill missing frames for specified track_id
        # track_id_list = [10]  # List of track_ids that need filling
        # filled_df = fill_missing_frames(df, track_id_list=track_id_list)
        filled_df = fill_missing_frames(df)

        # Save the filled data to a new Excel file
        filled_excel_save_path = save_dir / 'detections_filled.xlsx'
        filled_df.to_excel(filled_excel_save_path, index=False)
        print(f'填充后的数据已保存到 {filled_excel_save_path}')

    print(f'完成。({time.time() - t0:.3f}s)')


def fill_missing_frames(df, track_id_list=None):
    # Extract frame number, assuming image_name format is '000370.png'
    df['frame_number'] = df['image_name'].str.extract('(\d+)').astype(int)

    # Get all track_ids
    all_track_ids = df['track_id'].unique()

    # Store all filled data
    filled_df = pd.DataFrame()

    for track_id in all_track_ids:
        # Filter current track_id and sort by frame number
        track_df = df[df['track_id'] == track_id].sort_values('frame_number').reset_index(drop=True)

        # Create a new DataFrame to store filled data for current track_id
        track_filled_df = pd.DataFrame()

        # Check if current track_id is in specified track_id_list
        if track_id_list is None or track_id in track_id_list:
            # Fill missing frames for specified track_id
            for idx in range(len(track_df) - 1):
                current_row = track_df.loc[idx]
                next_row = track_df.loc[idx + 1]

                # Calculate difference between current frame and next frame
                frame_diff = next_row['frame_number'] - current_row['frame_number']

                # Add current row to filled data
                track_filled_df = track_filled_df.append(current_row, ignore_index=True)

                # If frame difference > 1, missing frames exist and need interpolation
                if frame_diff > 1:
                    for gap in range(1, frame_diff):
                        # Calculate interpolation ratio
                        ratio = gap / frame_diff

                        # Generate new frame number
                        new_frame_number = current_row['frame_number'] + gap
                        new_image_name = '{:06d}.png'.format(new_frame_number)

                        # Linearly interpolate x1, y1, x2, y2, score
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

                        # Add interpolated row to filled data
                        track_filled_df = track_filled_df.append(interpolated_row, ignore_index=True)
            # Add the last row
            track_filled_df = track_filled_df.append(track_df.iloc[-1], ignore_index=True)
        else:
            # For unspecified track_ids, directly add original data
            track_filled_df = track_df.copy()

        # Add current track_id filled data to total DataFrame
        filled_df = filled_df.append(track_filled_df, ignore_index=True)

    # Sort filled data by track_id and frame_number
    filled_df = filled_df.sort_values(['track_id', 'frame_number']).reset_index(drop=True)

    # Drop auxiliary frame_number column
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
                # Create a separate save directory for each subfolder
                save_dir = Path(opt.project) / opt.name / os.path.basename(subdir)
                save_dir = Path(increment_path(save_dir, exist_ok=opt.exist_ok))
                (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)
                # Call detect()
                detect(source=source_subdir, save_dir=save_dir)
        else:
            # If there are no subfolders, process the source directory
            save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
            (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)
            detect(source=source, save_dir=save_dir)
    else:
        # If source is not a directory, process directly
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)
        detect(source=source, save_dir=save_dir)


if __name__ == '__main__':
    from argparse import Namespace

    opt = Namespace(
        weights='pretrained/yolov7-e6e.pt',
        source='/media/robert/4TB-SSD/watchped_dataset - 副本/Images/sunny 2',  # Modify to your dataset path
        #img_size=1920,
        img_size=2560,
        #img_size=3840,
        conf_thres=0.03,
        iou_thres=0.7,
        # iou_thres=0.9,
        device='',  # Use default device
        view_img=False,
        save_txt=False,
        save_conf=False,
        nosave=False,
        classes=[0],  # Track only objects of class 0
        agnostic_nms=True,
        #augment=False,
        augment=True,
        update=False,
        project='/media/robert/4TB-SSD/watchped_dataset - 副本/Excel_Box',
        name='',
        exist_ok=False,
        trace=False,
        hide_labels_name=False,
        # Tracking parameters
        track_high_thresh=0.3,
        track_low_thresh=0.05,
        new_track_thresh=0.4,
        # track_high_thresh=0.1,
        # track_low_thresh=0.01,
        # new_track_thresh=0.2,
        track_buffer=5,  # pixels
        match_thresh=0.75,
        aspect_ratio_thresh=1.6,
        min_box_area=1,
        mot20=True,  # corresponds to --fuse-score
        # CMC
        cmc_method='sparseOptFlow',
        # ReID
        with_reid=True,  # Set to False if not using the ReID module
        fast_reid_config='fast_reid/configs/MOT20/sbs_S50.yml',
        fast_reid_weights='pretrained/mot20_sbs_S50.pth',
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        # Other options
        jde=False,
        ablation=False,
    )
    # Dynamically set opt.name to be consistent with the last directory name of opt.source
    opt.name = os.path.basename(opt.source.rstrip('/\\'))
    print(opt)
    with torch.no_grad():
        main()
