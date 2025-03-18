"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
import requests
import pygame

# Khởi tạo pygame mixer
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(r"C:\Users\vhoa1\Downloads\day di ong chau oi.mp3")

ESP_IP = "http://172.20.10.3"  # Thay bằng IP của ESP8266
is_playing = False  # Biến kiểm soát trạng thái âm thanh
start_time = None  # Theo dõi thời gian cảnh báo
normal_duration = 0  # Tổng thời gian ở trạng thái "normal"
abnormal_duration = 0  # Tổng thời gian ở trạng thái khác "normal"
last_update_time = time.time()  # Thời điểm cập nhật gần nhất
def reset_timers():
    global normal_duration, abnormal_duration, start_time, is_playing
    normal_duration = 0
    abnormal_duration = 0
    start_time = None
    is_playing = False


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

color_move_on = (255, 200, 90)
color_red = (25, 20, 240)
color = color_move_on
text_x_align = 10
inference_time_y = 30
fps_y = 90
analysis_time_y = 60
font_scale = 0.7
thickness = 2
rect_thickness = 2
tl = rect_thickness
total_arr = []
counter2 = 0
counter = 0


@torch.no_grad()  # Tắt tính toán gradient để tăng tốc và giảm sử dụng bộ nhớ
def run(weights=ROOT / 'weights/best.pt',  # Đường dẫn đến mô hình đã huấn luyện (file .pt)
        source=ROOT / 'data/images',  # Đầu vào: ảnh, video, webcam, hoặc URL luồng trực tiếp
        imgsz=640,  # Kích thước ảnh đầu vào (pixel)
        conf_thres=0.25,  # Ngưỡng tin cậy (Confidence threshold) để lọc dự đoán có độ tin cậy thấp
        iou_thres=0.45,  # Ngưỡng IoU (Intersection over Union) cho Non-Maximum Suppression (NMS)
        max_det=1000,  # Số lượng đối tượng tối đa được phát hiện trên mỗi ảnh
        device='',  # Chọn thiết bị: 'cpu' hoặc 'cuda:0' (nếu có GPU)
        view_img=False,  # Hiển thị kết quả dự đoán (True: bật, False: tắt)
        save_txt=False,  # Lưu kết quả vào file .txt
        save_conf=False,  # Lưu độ tin cậy của dự đoán vào file .txt
        save_crop=False,  # Lưu hình ảnh đã cắt từ hộp giới hạn của dự đoán
        nosave=False,  # Không lưu ảnh/video đầu ra nếu đặt thành True
        classes=None,  # Lọc theo lớp đối tượng (VD: --class 0 hoặc --class 0 2 3)
        agnostic_nms=False,  # NMS không phân biệt lớp (True: loại bỏ chồng lấn giữa các lớp khác nhau)
        augment=False,  # Dùng tăng cường dữ liệu (augmentation) khi chạy mô hình
        visualize=False,  # Hiển thị đặc trưng của các lớp trong mô hình
        update=False,  # Cập nhật mô hình
        project=ROOT / 'runs/detect',  # Thư mục lưu kết quả
        name='exp',  # Tên thư mục kết quả (VD: runs/detect/exp)
        exist_ok=False,  # Nếu thư mục kết quả đã tồn tại, không tạo mới
        line_thickness=3,  # Độ dày của đường viền hộp giới hạn (Bounding Box)
        hide_labels=False,  # Ẩn nhãn của các đối tượng phát hiện
        hide_conf=False,  # Ẩn độ tin cậy của dự đoán
        ):
    global counter2
    global counter
    global total_arr
    global is_playing, start_time, normal_duration, abnormal_duration, last_update_time
    alert_sound = pygame.mixer.Sound(r"C:\Users\vhoa1\Downloads\day di ong chau oi.mp3")

    is_playing = False  # 🚀 Biến kiểm soát trạng thái âm thanh
    normal_duration = 0  # 🔥 Tổng thời gian tài xế ở trạng thái "normal"
    last_update_time = time.time()  # 🔄 Lưu lại thời điểm cập nhật gần nhất
    abnormal_duration = 0  # 🔥 Tổng thời gian trạng thái bất thường (drowsy, yawning)
    normal_duration = 0  # 🔥 Tổng thời gian trạng thái normal trong 5 giây gần nhất
    last_update_time = time.time()  # 🔄 Lưu lại thời điểm cập nhật gần nhất
    start_time = None  # Đảm bảo biến được khởi tạo
    length = 0
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
            
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]

            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            normal_arr = []
            drowsy_arr = []
            drowsy2_arr = []
            yawning_arr = []

            # normal_num = 0
            # drowsy_num = 0
            # drowsy2_num = 0

            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                color = (255, 200, 90)
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Xử lý hiển thị bbox
                        c = int(cls)  # Lấy class của đối tượng
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                last_update_time = time.time()
    
                for path, img, im0s, vid_cap in dataset:
                    t_now = time.time()
                    elapsed = t_now - last_update_time
                    last_update_time = t_now
                    
                    img = torch.from_numpy(img).to(device).float()
                    pred = model(img, augment=augment)[0]

                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                    
                    for det in pred:
                        detected_state = "normal"
                        for *xyxy, conf, cls in reversed(det):
                            if names[int(cls)] in ['drowsy', 'drowsy#2', 'yawning']:
                                detected_state = names[int(cls)]
                                break
                        
                        if detected_state == "normal":
                            normal_duration += elapsed
                            abnormal_duration = max(abnormal_duration - elapsed, 0)
                        else:
                            abnormal_duration += elapsed
                            normal_duration = max(normal_duration - elapsed, 0)
                        
                        # Nếu tổng thời gian bất thường trên 3 giây trong tổng 5 giây, phát cảnh báo
                        if abnormal_duration >= 3 and (normal_duration + abnormal_duration) <= 5 and not is_playing:
                            print("🚨 Cảnh báo! Tài xế buồn ngủ!")
                            alert_sound.play()
                            is_playing = True
                            start_time = time.time()
                            alert_start_time = time.time() 
                        # Nếu đã phát cảnh báo và tài xế có trạng thái "normal" đủ 2 giây, tắt âm thanh
                        if abnormal_duration >= 10 and not is_playing:
                            print(f"🚀 Gửi lệnh phun nước! (Thời gian cảnh báo: {time.time() - alert_start_time:.2f} giây, Normal: {normal_duration:.2f})")
                            try:
                                response = requests.get(f"http://{ESP_IP}/spray", timeout=5)
                                print(f"✅ Phản hồi từ ESP: {response.status_code}")
                            except Exception as e:
                                print(f"❌ Lỗi gửi lệnh: {e}")
                        if is_playing and normal_duration >= 2:
                            print("✅ Tắt cảnh báo!")
                            alert_sound.stop()
                            reset_timers()
                        # Nếu trạng thái bất thường vẫn kéo dài > 10s mà chưa đủ 2s "normal", gửi lệnh phun nước
                        
                       


                        # Nếu trạng thái bất thường lại vượt 3 giây sau khi reset, kích hoạt lại cảnh báo
                        if not is_playing and abnormal_duration >= 3:
                            print("🚨 Cảnh báo lại! Tài xế buồn ngủ!")
                            alert_sound.play()
                            is_playing = True
                            start_time = time.time()


                        # Gán màu hiển thị như cũ
                        if names[c] == 'drowsy' or names[c] == 'drowsy#2':
                            color = (0, 0, 255)
                        elif names[c] == 'yawning':
                            color = (51, 255, 255)


                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        cv2.rectangle(im0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                        if names[c]:
                            if names[c] == 'normal':
                                normal_arr.append([names[c]])
                                drowsy_text = 'NORMAL '
                                label_size, base_line = cv2.getTextSize(drowsy_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                        font_scale,
                                                                        thickness)
                                label_ymin = max(30, label_size[1] + 10)
                                cv2.rectangle(im0, (text_x_align, label_ymin - label_size[1] - 10),
                                              (text_x_align + label_size[0], label_ymin + base_line - 10),
                                              (255, 200, 90),
                                              cv2.FILLED)
                                cv2.rectangle(im0, (text_x_align - 2, label_ymin - label_size[1] - 12),
                                              (text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
                                cv2.putText(im0, drowsy_text, (text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                                            font_scale,
                                            (0, 0, 0),
                                            thickness,
                                            cv2.LINE_AA)
                            elif names[c] == 'drowsy':
                                drowsy_arr.append([names[c]])
                                s_img = cv2.imread("icon/warning.jpg")
                                s_img = cv2.resize(s_img, (72, 60), cv2.INTER_LINEAR)
                                im0[33:33 + s_img.shape[0], 10:10 + s_img.shape[1]] = s_img

                                drowsy_text = 'WARNING '
                                label_size, base_line = cv2.getTextSize(drowsy_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                        font_scale,
                                                                        thickness)
                                label_ymin = max(30, label_size[1] + 10)
                                cv2.rectangle(im0, (text_x_align, label_ymin - label_size[1] - 10),
                                              (text_x_align + label_size[0], label_ymin + base_line - 10),
                                              (0, 0, 255),
                                              cv2.FILLED)
                                cv2.rectangle(im0, (text_x_align - 2, label_ymin - label_size[1] - 12),
                                              (text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
                                cv2.putText(im0, drowsy_text, (text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                                            font_scale,
                                            (0, 0, 0),
                                            thickness,
                                            cv2.LINE_AA)

                            elif names[c] == 'drowsy#2':
                                drowsy2_arr.append([names[c]])
                                s_img = cv2.imread("icon/warning.jpg")
                                s_img = cv2.resize(s_img, (72, 60), cv2.INTER_LINEAR)
                                im0[33:33 + s_img.shape[0], 10:10 + s_img.shape[1]] = s_img

                                drowsy_text = 'WARNING '
                                label_size, base_line = cv2.getTextSize(drowsy_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                        font_scale,
                                                                        thickness)
                                label_ymin = max(30, label_size[1] + 10)
                                cv2.rectangle(im0, (text_x_align, label_ymin - label_size[1] - 10),
                                              (text_x_align + label_size[0], label_ymin + base_line - 10),
                                              (0, 0, 255),
                                              cv2.FILLED)
                                cv2.rectangle(im0, (text_x_align - 2, label_ymin - label_size[1] - 12),
                                              (text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
                                cv2.putText(im0, drowsy_text, (text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                                            font_scale,
                                            (0, 0, 0),
                                            thickness,
                                            cv2.LINE_AA)

                            elif names[c] == 'yawning':
                                yawning_arr.append([names[c]])
                                s_img = cv2.imread("icon/coffe2.jpg")
                                s_img = cv2.resize(s_img, (72, 60), cv2.INTER_LINEAR)
                                im0[33:33 + s_img.shape[0], 10:10 + s_img.shape[1]] = s_img

                                drowsy_text = 'DROWSY '
                                label_size, base_line = cv2.getTextSize(drowsy_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                        font_scale,
                                                                        thickness)
                                label_ymin = max(30, label_size[1] + 10)
                                cv2.rectangle(im0, (text_x_align, label_ymin - label_size[1] - 10),
                                              (text_x_align + label_size[0], label_ymin + base_line - 10),
                                              (51, 255, 255),
                                              cv2.FILLED)
                                cv2.rectangle(im0, (text_x_align - 2, label_ymin - label_size[1] - 12),
                                              (text_x_align + 2 + label_size[0], label_ymin + base_line - 8), (0, 0, 0))
                                cv2.putText(im0, drowsy_text, (text_x_align, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX,
                                            font_scale,
                                            (0, 0, 0),
                                            thickness,
                                            cv2.LINE_AA)

                            lab_text = '{} {:0.2f} '.format(names[c],conf)
                            tf = max(tl - 1, 1)  # font thickness
                            t_size = cv2.getTextSize(lab_text, 0, fontScale=tl / 3, thickness=tf)[0]
                            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                            cv2.rectangle(im0, c1, c2, color, -1, cv2.LINE_AA)  # filled
                            # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            cv2.putText(im0, lab_text, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf,
                                        lineType=cv2.LINE_AA)

                            normal_num = len(normal_arr)
                            drowsy_num = len(drowsy_arr)
                            drowsy2_num = len(drowsy2_arr)
                            yawning_num = len(yawning_arr)

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                # cv2.waitKey(1)  # 1 millisecond
                k = cv2.waitKey(1) & 0xFF  # Chạy video liên tục


                if k == 27:  # wsc
                    cv2.destroyAllWindows()

            # # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / './weights/yolov5l.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
