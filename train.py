from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

class Custom_DeepSORT_Tracker:
    def __init__(self):
       self.tracker = DeepSort(max_age=5)

    def __del__(self):
        pass

    def update_tracker(self, detections):
        bbs = []
        if detections[0].boxes.xywh.tolist():
            for d in detections: 
                box = d.boxes.xywh.tolist()[0]
                conf = d.boxes.conf.tolist()[0]
                classes = d.boxes.cls.tolist()[0]
                bb = (box, conf, classes)
                bbs.append(bb)

        # bbs expected to be a list of detections, each in tuples of 
        # ( [left,top,width,height], confidence, detection_class )
        tracks = self.tracker.update_tracks(bbs, frame=detections[0].orig_img)
        return tracks

print("Worker")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    video_source = r'./Videos/urea.mp4'

    tracker = Custom_DeepSORT_Tracker()

    model = YOLO('yolo11n-obb.pt')
    model.to(device)
    model.conf = 0.9  # confidence threshold (0-1)

    results = model.train(data="config.yaml", epochs=200, save_period=50, imgsz=640, device='cuda')

