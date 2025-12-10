from ultralytics import YOLO
import os
import shutil
import wandb
import argparse

def train():
    # Ask the user to log into Weights & Biases if not logged in
    wandb.login()
    # Initialize wandb project and disable system stats (GPU memory, etc.) as requested
    wandb.init(project="ppo-aimbot", settings=wandb.Settings(_disable_stats=True))

    parser = argparse.ArgumentParser(description='Train YOLOv11 model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    args = parser.parse_args()

    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Custom callback to save model every 10 epochs and override the same file
    def save_every_10_epochs(trainer):
        if (trainer.epoch + 1) % 10 == 0:
            weights_dir = trainer.save_dir / "weights"
            last_pt = weights_dir / "last.pt"
            # Define a constant filename to overwrite
            target_pt = weights_dir / "every_10_epochs.pt"
            
            if last_pt.exists():
                shutil.copy(last_pt, target_pt)
                print(f"\nSaved custom checkpoint to {target_pt}")

    model.add_callback("on_train_epoch_end", save_every_10_epochs)

    # Train the model
    # data.yaml path should be absolute to avoid CWD confusion
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "data.yaml"))
    
    print(f"Training with dataset config: {dataset_path}")
    
    results = model.train(data=dataset_path, epochs=args.epochs, imgsz=640)

if __name__ == "__main__":
    train()
