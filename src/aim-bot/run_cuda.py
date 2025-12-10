import time
import os
import random
import numpy as np
import mss
import cv2
import pyautogui
import torch
import socket
import json
import argparse
from ultralytics import YOLO
from brain_cuda import ActorCritic

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "aim-bot", "weights", "best_aim_cuda.pth")
YOLO_PATH = os.path.join(PROJECT_ROOT,"src", "yolo", "models", "best.pt")  # Adjust relative path if needed
CONF_THRESHOLD = 0.15
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CENTER_X, CENTER_Y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
DETECTION_BOX_SIZE = 640 
CAPTURE_REGION = {'top': CENTER_Y - 320, 'left': CENTER_X - 320, 'width': 640, 'height': 640}

DEBUG_DIR = os.path.join(PROJECT_ROOT, "debug_screenshots")
os.makedirs(DEBUG_DIR, exist_ok=True)

# Mouse control settings
MOUSE_SENSITIVITY = 5.0 # How many pixels to move per unit of action

def main():
    # 0. Device Setup
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Running inference on: {device}")

    # 1. Load Models
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-mode', action='store_true', help="Run in test mode with external coordinates")
    args = parser.parse_args()

    # 1. Load Models
    print("Loading YOLO...")
    if not args.test_mode:
        yolo = YOLO(YOLO_PATH)
    else:
        print("Test Mode: YOLO load skipped.")
    
    print("Loading Brain...")
    # Obs dim: 2, Action dim: 2
    brain = ActorCritic((2,), (2,)).to(device)
    brain.eval() # Set to eval mode
    
    if os.path.exists(MODEL_PATH):
        try:
            brain.load_weights(MODEL_PATH, device=device)
            print(f"Loaded weights from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Running with random weights.")
    else:
        print(f"Warning: Trained weights not found at {MODEL_PATH}! Running with random weights.")

    sct = mss.mss()
    
    if args.test_mode:
        print("Test Mode Active. Listening for coordinates on UDP 9999...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 9999))
        sock.setblocking(False)

    print("Aimbot Active. Press Ctrl+C to stop.")
    
    try:
        while True:
            target_box = None
            
            if args.test_mode:
                # 2a. Receive Coordinates (Test Mode)
                try:
                    data, addr = sock.recvfrom(1024)
                    data = json.loads(data.decode())
                    # Game coordinates (0,0 is top-left)
                    # We map them to match screen logic if needed, but here simple mapping
                    # Assume game center (320, 320) matches capture center
                    tx, ty = data['x'], data['y']
                    target_box = (tx, ty)
                    # No image capture needed
                except BlockingIOError:
                    pass # No new data
                except Exception as e:
                    print(f"UDP Error: {e}")
                
                time.sleep(0.01) # Small sleep to match ~60-100 FPS

            else:
                # 2b. Capture Screen (Normal Mode)
                img = np.array(sct.grab(CAPTURE_REGION))
                # Delete alpha channel
                img = img[:, :, :3]
                
                # 3. Detect
                results = yolo(img, verbose=False, conf=CONF_THRESHOLD)
                
                min_dist = float('inf')
                
                # 4. Find Nearest Target
                # Process results relative to the CAPTURE REGION center
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
                            target_box = (x, y) 
            
            # Common Logic: Act
            img_center_x, img_center_y = 320, 320 # Center of 640x640 region

            if target_box:
                tx, ty = target_box
                
                # Normalize relative position to -1 to 1 range for the brain
                # (rel_x, rel_y)
                rel_x = (tx - img_center_x) / 320.0
                rel_y = (ty - img_center_y) / 320.0
                
                obs = np.array([rel_x, rel_y], dtype=np.float32)
                
                # Query Brain
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                
                # get_action_mean return tensor, move to cpu numpy
                mean = brain.get_action_mean(obs_tensor)
                action = mean.detach().cpu().numpy().flatten()
                
                # 6. Move Mouse
                dx = action[0] * MOUSE_SENSITIVITY * 10 
                dy = action[1] * MOUSE_SENSITIVITY * 10 
                
                print(f"[DEBUG] Target: ({tx:.1f}, {ty:.1f}) | Rel: ({rel_x:.2f}, {rel_y:.2f}) | Action: ({action[0]:.3f}, {action[1]:.3f}) | Delta: ({dx:.1f}, {dy:.1f}) -> Move: ({int(dx)}, {int(dy)})")

                if args.test_mode:
                    # Send feedback to game instead of moving mouse
                    feedback = json.dumps({"dx": int(dx), "dy": int(dy)}).encode('utf-8')
                    # Send to localhost:9998 (simple_game listener)
                    sock.sendto(feedback, ("127.0.0.1", 9998))
                else:
                    pyautogui.moveRel(int(dx), int(dy))
                
            # --- DEBUG: Save Screenshot ---
            # Save if we found a target OR with some small random probability
            if not args.test_mode and (target_box is not None or random.random() < 0.05):
                # Plot results on the image (this returns BGR numpy array usually)
                annotated_frame = results[0].plot()
                
                # Make filename unique
                timestamp = int(time.time() * 1000)
                filename = os.path.join(DEBUG_DIR, f"debug_{timestamp}.jpg")
                
                # Save using OpenCV
                # annotated_frame from ultralytics plot() is usually BGR
                cv2.imwrite(filename, annotated_frame)
            # time.sleep(0.01) 
            
    except KeyboardInterrupt:
        print("Stopping.")

if __name__ == "__main__":
    main()
