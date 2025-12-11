import time
import os
import numpy as np
import pyautogui
import torch
from pynput import mouse
from brain_cuda import ActorCritic

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "aim-bot", "weights", "best_aim_cuda.pth")
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CENTER_X, CENTER_Y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

# Mouse control settings
MOUSE_SENSITIVITY = 5.0
DELAY_SECONDS = 5
DISTANCE_THRESHOLD = 10  # Stop when within 10 pixels of target
MAX_ITERATIONS = 100  # Safety limit to prevent infinite loops
STEP_DELAY = 0.01  # Small delay between movements for smoothness

# Global variable to store clicked position
target_position = None

def on_click(x, y, button, pressed):
    """Callback for mouse clicks"""
    global target_position
    
    if pressed and button == mouse.Button.left:
        target_position = (x, y)
        print(f"Target set to ({x}, {y}). Moving in {DELAY_SECONDS} seconds...")
        return False  # Stop listener after first click

def wait_for_click():
    """Wait for user to click and return the clicked position"""
    global target_position
    target_position = None
    
    print("Click anywhere on screen to set target...")
    
    # Start listening for mouse clicks
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    
    return target_position

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def move_step_to_target(brain, device, target_x, target_y):
    """Make one step toward the target using the brain"""
    current_x, current_y = pyautogui.position()
    
    # Calculate distance to target
    distance = calculate_distance(current_x, current_y, target_x, target_y)
    
    # Calculate relative position to the capture region center
    img_center_x, img_center_y = 320, 320  # Center of 640x640 region
    
    # Convert screen coordinates to capture region coordinates
    # Target relative to screen center
    rel_to_center_x = target_x - CENTER_X
    rel_to_center_y = target_y - CENTER_Y
    
    # Now map to capture region coordinates (320, 320)
    tx = img_center_x + rel_to_center_x
    ty = img_center_y + rel_to_center_y
    
    # Normalize relative position to -1 to 1 range for the brain
    rel_x = (tx - img_center_x) / 320.0
    rel_y = (ty - img_center_y) / 320.0
    
    obs = np.array([rel_x, rel_y], dtype=np.float32)
    
    # Query Brain
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    
    # Get action mean
    mean = brain.get_action_mean(obs_tensor)
    action = mean.detach().cpu().numpy().flatten()
    
    # Calculate movement
    dx = action[0] * MOUSE_SENSITIVITY * 10
    dy = action[1] * MOUSE_SENSITIVITY * 10
    
    print(f"[STEP] Dist: {distance:.1f}px | Target: ({tx:.1f}, {ty:.1f}) | Rel: ({rel_x:.2f}, {rel_y:.2f}) | Action: ({action[0]:.3f}, {action[1]:.3f}) | Move: ({int(dx)}, {int(dy)})")
    
    # Move mouse
    pyautogui.moveRel(int(dx), int(dy))
    
    return distance

def move_to_target_smoothly(brain, device, target_x, target_y):
    """Continuously move toward target until close enough"""
    iteration = 0
    
    while iteration < MAX_ITERATIONS:
        current_x, current_y = pyautogui.position()
        distance = calculate_distance(current_x, current_y, target_x, target_y)
        
        # Check if we've reached the target
        if distance < DISTANCE_THRESHOLD:
            print(f"[SUCCESS] Reached target! Final distance: {distance:.2f}px")
            break
        
        # Make one step toward target
        move_step_to_target(brain, device, target_x, target_y)
        
        iteration += 1
        time.sleep(STEP_DELAY)
    
    if iteration >= MAX_ITERATIONS:
        print(f"[TIMEOUT] Stopped after {MAX_ITERATIONS} iterations. Final distance: {distance:.2f}px")

def main():
    # 0. Print Screen Resolution
    print(f"Screen Resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print(f"Screen Center: ({CENTER_X}, {CENTER_Y})")
    
    # 1. Device Setup
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Running inference on: {device}")
    
    # 1. Load Brain
    print("Loading Brain...")
    brain = ActorCritic((2,), (2,)).to(device)
    brain.eval()  # Set to eval mode
    
    if os.path.exists(MODEL_PATH):
        try:
            brain.load_weights(MODEL_PATH, device=device)
            print(f"Loaded weights from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Running with random weights.")
    else:
        print(f"Warning: Trained weights not found at {MODEL_PATH}! Running with random weights.")
    
    print("\nClick-to-Aim Bot Active. Press Ctrl+C to stop.")
    print(f"Settings: Distance threshold = {DISTANCE_THRESHOLD}px, Max iterations = {MAX_ITERATIONS}")
    
    try:
        while True:
            # Wait for user click
            target_pos = wait_for_click()
            
            if target_pos:
                target_x, target_y = target_pos
                
                # Wait 5 seconds
                print(f"\nWaiting {DELAY_SECONDS} seconds...")
                time.sleep(DELAY_SECONDS)
                
                # Move mouse to target smoothly
                print(f"Moving toward target ({target_x}, {target_y})...\n")
                move_to_target_smoothly(brain, device, target_x, target_y)
                
                print("\nReady for next target.\n")
            
    except KeyboardInterrupt:
        print("\nStopping.")

if __name__ == "__main__":
    main()