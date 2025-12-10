import time
import os
import numpy as np
import mss
import pyautogui
import mlx.core as mx
from ultralytics import YOLO
from brain import ActorCritic

# --- Config ---
MODEL_PATH = "weights/best_aim.npz"
YOLO_PATH = "../../best.pt"  # Adjust relative path if needed
CONF_THRESHOLD = 0.5
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CENTER_X, CENTER_Y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
DETECTION_BOX_SIZE = 640 # Capture a simplified box around center or full screen?
# For speed, let's capture a region around the center.
CAPTURE_REGION = {'top': CENTER_Y - 320, 'left': CENTER_X - 320, 'width': 640, 'height': 640}

# Mouse control settings
MOUSE_SENSITIVITY = 5.0 # How many pixels to move per unit of action

def main():
    # 1. Load Models
    print("Loading YOLO...")
    yolo = YOLO(YOLO_PATH)
    
    print("Loading Brain...")
    # Obs dim: 2, Action dim: 2
    brain = ActorCritic((2,), (2,))
    if os.path.exists(MODEL_PATH):
        brain.load_weights(MODEL_PATH)
    else:
        print("Warning: Trained weights not found! Running with random weights.")

    sct = mss.mss()
    
    print("Aimbot Active. Press Ctrl+C to stop.")
    
    try:
        while True:
            # 2. Capture Screen
            img = np.array(sct.grab(CAPTURE_REGION))
            # Delete alpha channel
            img = img[:, :, :3]
            
            # 3. Detect
            results = yolo(img, verbose=False, conf=CONF_THRESHOLD)
            
            target_box = None
            min_dist = float('inf')
            
            # 4. Find Nearest Target
            # Process results relative to the CAPTURE REGION center (which corresponds to screen center)
            # Capture region center is (320, 320) in the image coords.
            img_center_x, img_center_y = 320, 320
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Box xywh
                    x, y, w, h = box.xywh[0] # Tensor
                    x, y = float(x), float(y)
                    
                    # Distance from center
                    dist = np.sqrt((x - img_center_x)**2 + (y - img_center_y)**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        target_box = (x, y) # This is absolute position in the IMAGE
                        
            # 5. Act
            if target_box:
                tx, ty = target_box
                
                # Normalize relative position to -1 to 1 range for the brain
                # (rel_x, rel_y)
                # Max distance is approx 320 pixels.
                rel_x = (tx - img_center_x) / 320.0
                rel_y = (ty - img_center_y) / 320.0
                
                obs = np.array([rel_x, rel_y], dtype=np.float32)
                
                # Query Brain (get mean action, no noise needed for inference)
                # We can use the helper or just call the model.
                mean, _, _ = brain(mx.array(obs[None, :]))
                action = np.array(mean).flatten()
                
                # 6. Move Mouse
                # Action is -1 to 1 (approx). Scale to pixels.
                dx = action[0] * MOUSE_SENSITIVITY * 10 # Multiplier
                dy = action[1] * MOUSE_SENSITIVITY * 10 
                
                pyautogui.moveRel(int(dx), int(dy))
                
            # Sleep slightly to prevent CPU hogging if needed, but we want max FPS.
            # time.sleep(0.01) 
            
    except KeyboardInterrupt:
        print("Stopping.")

if __name__ == "__main__":
    main()
