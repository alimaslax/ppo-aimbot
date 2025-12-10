from ultralytics import YOLO
import os

import argparse

def train():
    parser = argparse.ArgumentParser(description='Train YOLOv11 model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    args = parser.parse_args()

    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    # data.yaml path should be absolute to avoid CWD confusion
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "data.yaml"))
    
    print(f"Training with dataset config: {dataset_path}")
    
    results = model.train(data=dataset_path, epochs=args.epochs, imgsz=640)

if __name__ == "__main__":
    train()
