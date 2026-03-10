from model_loader import get_model


def main():
    model = get_model()

    model.train(
        data="dataset/data.yaml",
        epochs=10,
        batch=16,
        imgsz=640,
        device=0,
    )


if __name__ == "__main__":
    main()
