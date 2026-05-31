from ultralytics import YOLO


def main():
    model = YOLO("yolov8n-seg.pt")

    model.train(
        data="data.yaml",
        epochs=10,
        imgsz=512,
        batch=4,
        device="cpu",
        project="runs",
        name="crack_yolov8n_seg_10ep",
        patience=5,
        workers=0
    )


if __name__ == "__main__":
    main()