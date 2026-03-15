from ultralytics import YOLO
import torch

def main():

    model = YOLO("runs/segment/train/weights/last.pt")

    model.train(
        data='dataset/data.yaml',
        epochs=100,
        batch=0.9,
        imgsz=780,
        device=0,
        dropout=0.2
    )

if __name__ == "__main__":
    main()