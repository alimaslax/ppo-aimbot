from ultralytics import YOLO
import os

def test():
    # Load the best model
    model_path = os.path.abspath("best.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = YOLO(model_path)

    # Path to validation image
    image_path = os.path.abspath("validate/1.png")
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Run inference with lower confidence threshold
    print(f"Running inference on {image_path} with conf=0.05...")
    results = model(image_path, conf=0.05)

    # Visualize and save results
    for result in results:
        print(f"Detected {len(result.boxes)} objects.")
        if len(result.boxes) > 0:
            for box in result.boxes:
                print(f"Class: {int(box.cls)}, Conf: {float(box.conf):.4f}")
        
        result.save(filename="prediction_result.jpg")  # save to disk
        print("Prediction saved to 'prediction_result.jpg'")
        result.show()  # show to screen

if __name__ == "__main__":
    test()
