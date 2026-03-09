from ultralytics import YOLO

def main():

    model = YOLO("yolo26l-seg.pt")

    model.train(
        data='data.yaml',
        epochs=10,
        batch=16,
        imgsz=640,
        device=0
    )

if __name__ == "__main__":
    main()