from ultralytics import YOLO

model = YOLO(
    "runs/segment/runs/crack_yolov8n_seg_10ep/weights/best.pt"
)

metrics = model.val()

print(metrics)