from ultralytics import YOLO
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    video_source = r'./Videos/video4.mp4'

    model = YOLO('yolo11n-obb.pt')
    model.to(device)
    model.conf = 0.9  # confidence threshold (0-1)

    results = model.train(data="config.yaml", epochs=200, save_period=50, imgsz=640, device='cuda')

